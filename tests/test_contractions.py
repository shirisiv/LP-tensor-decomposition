import numpy as np

from lp_tensor_decomposition import alternating_contraction
from lp_tensor_decomposition.contractions import complementary_minor_matrix


def test_complementary_minor_matrix_has_rank_at_most_two():
    rng = np.random.default_rng(0)
    for m in range(3, 8):
        M = rng.standard_normal((m - 2, m))
        K = complementary_minor_matrix(M)
        assert np.linalg.matrix_rank(K, tol=1e-9) == 2
        assert np.linalg.norm(M @ K) < 1e-9


def test_three_way_contraction_matches_block_formula():
    rng = np.random.default_rng(1)
    T = rng.standard_normal((2, 3, 2))
    D = sum(T.shape)
    p = rng.standard_normal(D)
    Omega = alternating_contraction(T, p[None, :])

    a = p[:2]
    b = p[2:5]
    c = p[5:]

    T_gamma = np.tensordot(T, c, axes=(2, 0))
    T_beta = np.tensordot(T, b, axes=(1, 0))
    # tensordot leaves axes (A,C), as needed.
    T_alpha = np.tensordot(a, T, axes=(0, 0))

    expected = np.zeros((D, D))
    expected[:2, 2:5] = T_gamma
    expected[:2, 5:] = -T_beta
    expected[2:5, 5:] = T_alpha
    expected = expected - expected.T

    assert np.allclose(Omega, expected, atol=1e-10)
