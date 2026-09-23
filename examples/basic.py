import numpy as np

from lp_tensor_decomposition import cp_tensor, decompose_lp

rng = np.random.default_rng(0)
r = 3
factors = [rng.standard_normal((3, r)) for _ in range(3)]
T = cp_tensor(factors)

result = decompose_lp(T, rank=r, random_state=1)
print("rank:", result.rank)
print("relative reconstruction residual:", result.residual)
print("spectral values:", result.eigenvalues)
