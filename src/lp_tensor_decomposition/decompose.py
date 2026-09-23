from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg

from .contractions import alternating_contraction, block_offsets
from .tensor_utils import (
    compress_tensor,
    cp_tensor,
    numerical_rank,
    rank_one_tensor,
    relative_residual,
)


@dataclass
class LPDecompositionResult:
    factors: list[NDArray[np.float64]]
    rank: int
    residual: float
    P: NDArray[np.float64]
    Q: NDArray[np.float64]
    omega_P: NDArray[np.float64]
    omega_Q: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    component_matrices: list[NDArray[np.float64]]
    compressed_shape: tuple[int, ...]
    compression_bases: list[NDArray[np.float64]] | None


def _svd_rank_and_image_basis(
    A: NDArray[np.float64],
    *,
    rtol: float,
    atol: float,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64]]:
    """Return numerical rank, an orthonormal image basis, and a kernel basis."""
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        rank = 0
    else:
        rank = int(np.count_nonzero(s > atol + rtol * s[0]))
    image = U[:, :rank]
    kernel = Vh.T[:, rank:]
    return rank, image, kernel


def _pair_repeated_eigenvalues(
    Phi: NDArray[np.float64],
    r: int,
    *,
    imag_tol: float,
    pair_tol: float,
) -> NDArray[np.float64]:
    vals = linalg.eigvals(Phi)
    scale = max(1.0, float(np.max(np.abs(vals))))
    if np.max(np.abs(vals.imag)) > imag_tol * scale:
        raise np.linalg.LinAlgError("spectral step produced non-real eigenvalues")
    vals = vals.real
    vals = np.sort(vals)
    if vals.size != 2 * r:
        raise np.linalg.LinAlgError("unexpected number of eigenvalues")

    paired = vals.reshape(r, 2)
    within = np.abs(paired[:, 0] - paired[:, 1])
    centers = paired.mean(axis=1)

    if r > 1:
        separation = np.min(np.diff(np.sort(centers)))
        threshold = pair_tol * max(1.0, float(np.max(np.abs(centers))))
        # Each repeated pair should be much tighter than the separation between
        # distinct eigenvalues.  If not, retry with a different random q.
        if np.max(within) > max(threshold, 0.1 * separation):
            raise np.linalg.LinAlgError("could not reliably identify double eigenvalues")
    elif np.max(within) > pair_tol * max(1.0, abs(float(centers[0]))):
        raise np.linalg.LinAlgError("could not reliably identify the repeated eigenvalue")

    if np.any(np.abs(centers) <= pair_tol * max(1.0, float(np.max(np.abs(centers))))):
        raise np.linalg.LinAlgError("spectral step produced a numerically zero eigenvalue")
    return centers


def _spectral_projectors(
    Phi: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    n = Phi.shape[0]
    I = np.eye(n)
    projectors: list[NDArray[np.float64]] = []
    for i, rho_i in enumerate(eigenvalues):
        Pi = np.eye(n)
        for j, rho_j in enumerate(eigenvalues):
            if i == j:
                continue
            Pi = Pi @ ((Phi - rho_j * I) / (rho_i - rho_j))
        projectors.append(Pi)
    return projectors


def _factor_component_blocks(
    omega: NDArray[np.float64],
    shape: Sequence[int],
    *,
    block_tol: float,
) -> list[NDArray[np.float64]]:
    """Recover one representative from each factor line using blocks (1,j)."""
    offsets = block_offsets(shape)
    m = len(shape)
    vectors: list[NDArray[np.float64] | None] = [None] * m

    for j in range(1, m):
        block = omega[offsets[0] : offsets[1], offsets[j] : offsets[j + 1]]
        U, s, Vh = np.linalg.svd(block, full_matrices=False)
        if s.size == 0 or s[0] <= block_tol:
            raise np.linalg.LinAlgError(
                f"component block (1,{j+1}) is numerically zero; retry random contraction"
            )
        if vectors[0] is None:
            vectors[0] = U[:, 0]
        vectors[j] = Vh[0, :]

    assert all(v is not None for v in vectors)
    return [np.asarray(v, dtype=float) for v in vectors]  # type: ignore[arg-type]


def _solve_component_weights(
    tensor: NDArray[np.float64],
    component_vectors: list[list[NDArray[np.float64]]],
) -> NDArray[np.float64]:
    columns = [rank_one_tensor(vs).reshape(-1) for vs in component_vectors]
    design = np.column_stack(columns)
    coeffs, _, _, _ = np.linalg.lstsq(design, tensor.reshape(-1), rcond=None)
    return coeffs


def decompose_lp(
    tensor: ArrayLike,
    *,
    rank: int | None = None,
    compressed: bool = True,
    random_state: int | np.random.Generator | None = None,
    max_retries: int = 20,
    rank_rtol: float = 1e-9,
    rank_atol: float = 0.0,
    eig_imag_tol: float = 1e-7,
    eig_pair_tol: float = 1e-6,
    block_tol: float = 1e-10,
) -> LPDecompositionResult:
    """Decompose a tensor satisfying the Lovitz--Petrov condition.

    Parameters
    ----------
    tensor:
        Dense real tensor of order m >= 3.
    rank:
        Optional CP rank.  If omitted, it is inferred generically as
        rank(Omega_P)/2.
    compressed:
        If True (default), `tensor` is assumed to already be in the compressed
        form used in the paper.  If False, each mode is first orthogonally
        compressed using an SVD and the recovered factors are lifted back.
    random_state:
        Seed or NumPy Generator used for the generic contractions.
    max_retries:
        Number of random contractions attempted before declaring numerical
        failure.

    Notes
    -----
    The theorem is exact/algebraic and succeeds with probability one for
    absolutely continuous random contractions.  This routine is a floating-
    point reference implementation, so ill-conditioning can require retries or
    adjusted tolerances.
    """
    T_input = np.asarray(tensor, dtype=float)
    if T_input.ndim < 3:
        raise ValueError("tensor order must be at least 3")

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    compression_bases: list[NDArray[np.float64]] | None = None
    if compressed:
        T = T_input
    else:
        T, compression_bases = compress_tensor(T_input, rtol=rank_rtol, atol=rank_atol)

    shape = T.shape
    m = T.ndim
    d = m - 2
    D = int(sum(shape))

    last_error: Exception | None = None

    for _ in range(max_retries):
        try:
            # Step 1: generic P and Omega_P.
            P = rng.standard_normal((d, D))
            OmegaP = alternating_contraction(T, P)

            omega_rank, image_basis, kernel_basis = _svd_rank_and_image_basis(
                OmegaP, rtol=rank_rtol, atol=rank_atol
            )
            if omega_rank == 0 or omega_rank % 2:
                raise np.linalg.LinAlgError("Omega_P does not have positive even numerical rank")

            inferred_rank = omega_rank // 2
            if rank is not None and inferred_rank != rank:
                raise np.linalg.LinAlgError(
                    f"rank(Omega_P)/2 = {inferred_rank}, expected rank={rank}"
                )
            r = inferred_rank if rank is None else rank

            if kernel_basis.shape[1] == 0:
                raise np.linalg.LinAlgError("Omega_P has trivial kernel")

            # Step 2: generic q in ker Omega_P and Q=(q,s_1,...,s_{m-3}).
            coeff = rng.standard_normal(kernel_basis.shape[1])
            q = kernel_basis @ coeff
            qnorm = np.linalg.norm(q)
            if qnorm == 0:
                raise np.linalg.LinAlgError("sampled the zero kernel vector")
            q /= qnorm
            Q = P.copy()
            Q[0, :] = q
            OmegaQ = alternating_contraction(T, Q)

            # Step 3: rows of U form an orthonormal basis of Im Omega_P.
            Urows = image_basis.T

            # Step 4: restriction to L and Phi = OmegaQ~ OmegaP~^{-1}.
            OmegaP_tilde = Urows @ OmegaP @ Urows.T
            OmegaQ_tilde = Urows @ OmegaQ @ Urows.T
            if numerical_rank(OmegaP_tilde, rtol=rank_rtol, atol=rank_atol) != 2 * r:
                raise np.linalg.LinAlgError("restricted Omega_P is numerically singular")
            Phi = linalg.solve(OmegaP_tilde.T, OmegaQ_tilde.T, assume_a="gen").T

            # Step 5: r distinct eigenvalues, each with multiplicity two.
            eigenvalues = _pair_repeated_eigenvalues(
                Phi, r, imag_tol=eig_imag_tol, pair_tol=eig_pair_tol
            )
            projectors = _spectral_projectors(Phi, eigenvalues)

            # Step 6: isolate the component matrices omega_i.
            component_matrices: list[NDArray[np.float64]] = []
            for Pi in projectors:
                omega = Urows.T @ Pi @ OmegaP_tilde @ Urows
                omega = np.real_if_close(omega, tol=1000).astype(float)
                omega = 0.5 * (omega - omega.T)
                component_matrices.append(omega)

            # Step 7: recover one vector on each factor line from blocks (1,j).
            components = [
                _factor_component_blocks(omega, shape, block_tol=block_tol)
                for omega in component_matrices
            ]

            # Steps 8--9: solve for component weights and absorb them into mode 1.
            weights = _solve_component_weights(T, components)
            for i, lam in enumerate(weights):
                components[i][0] = lam * components[i][0]

            factors_core = [
                np.column_stack([components[i][mode] for i in range(r)])
                for mode in range(m)
            ]

            if compression_bases is None:
                factors = factors_core
            else:
                factors = [B @ A for B, A in zip(compression_bases, factors_core)]

            residual = relative_residual(T_input, factors)
            return LPDecompositionResult(
                factors=factors,
                rank=r,
                residual=residual,
                P=P,
                Q=Q,
                omega_P=OmegaP,
                omega_Q=OmegaQ,
                eigenvalues=eigenvalues,
                component_matrices=component_matrices,
                compressed_shape=tuple(shape),
                compression_bases=compression_bases,
            )

        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"LP decomposition failed after {max_retries} random attempts; "
        f"last numerical failure: {last_error}"
    )
