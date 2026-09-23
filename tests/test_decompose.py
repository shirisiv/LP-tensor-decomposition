import numpy as np

from lp_tensor_decomposition import cp_tensor, decompose_lp


def test_three_way_generic_lp_tensor():
    rng = np.random.default_rng(2)
    r = 3
    factors = [rng.standard_normal((3, r)) for _ in range(3)]
    T = cp_tensor(factors)

    result = decompose_lp(T, rank=r, random_state=3, max_retries=10)
    assert result.rank == r
    assert result.residual < 1e-7


def test_four_way_generic_lp_tensor():
    rng = np.random.default_rng(4)
    r = 3
    factors = [rng.standard_normal((3, r)) for _ in range(4)]
    T = cp_tensor(factors)

    result = decompose_lp(T, rank=r, random_state=5, max_retries=10)
    assert result.rank == r
    assert result.residual < 1e-7


def test_lp_tensor_beyond_kruskal():
    """An LP example that does not satisfy Kruskal's condition."""
    # A has k-rank 2 (columns 1,2,3 are dependent but every pair is independent).
    A = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # B is a 3-by-4 Vandermonde matrix, so every triple is independent: k_B=3.
    t = np.array([0.0, 1.0, 2.0, 3.0])
    B = np.vstack([np.ones(4), t, t**2])
    # C is invertible: k_C=4.  Hence 2+3+4=9 < 2r+2=10,
    # so Kruskal's condition fails.  The LP inequalities nevertheless hold.
    C = np.eye(4)
    T = cp_tensor([A, B, C])

    result = decompose_lp(T, rank=4, random_state=8, max_retries=20)
    assert result.rank == 4
    assert result.residual < 1e-7
