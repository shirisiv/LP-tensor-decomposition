# Lovitz--Petrov tensor decomposition

Reference Python implementation of the spectral tensor-decomposition algorithm under the Lovitz--Petrov (LP) identifiability condition.

The implementation follows the paper notation closely.  Given a compressed real tensor

\[
\mathcal T\in\mathbb R^{r_1\times\cdots\times r_m},
\]

it constructs the alternating contractions \(\Omega_P,\Omega_Q\), restricts them to \(L=\operatorname{im}\Omega_P\), separates the two-dimensional component spaces by the spectrum of

\[
\Phi=\widetilde\Omega_Q\widetilde\Omega_P^{-1},
\]

recovers the component matrices \(\omega_i\), factors their rank-one blocks, and solves for the final component scalings.

## Status

This is a **floating-point reference implementation** of an exact algebraic algorithm.  The theorem succeeds with probability one for generic contractions.  Numerically, conditioning matters; the implementation therefore retries generic contractions when the spectral structure is not resolved cleanly.

## Installation

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[dev]'
pytest
```

## Minimal example

```python
import numpy as np
from lp_tensor_decomposition import cp_tensor, decompose_lp

rng = np.random.default_rng(0)
r = 3
A = rng.standard_normal((3, r))
B = rng.standard_normal((3, r))
C = rng.standard_normal((3, r))
T = cp_tensor([A, B, C])

result = decompose_lp(T, rank=r, random_state=1)
print(result.residual)
recovered = result.factors
```

The recovered factors are equivalent to the original ones up to a common permutation of the columns and componentwise scaling.

## API

### `decompose_lp(tensor, ...)`

By default, `tensor` is assumed to be in the compressed form used in the paper.  The CP rank may be supplied with `rank=...`; if omitted, the code infers it from

\[
r=\tfrac12\operatorname{rank}(\Omega_P).
\]

Set `compressed=False` to perform a numerical orthogonal mode compression before running the LP algorithm, then lift the recovered factors back to the original mode spaces.

The returned `LPDecompositionResult` contains the factor matrices as well as the intermediate objects `P`, `Q`, `omega_P`, `omega_Q`, the recovered spectral values, and the individual component matrices.  These diagnostics are intentionally exposed so experiments can be compared directly with the paper's lemmas.

## Construction of `Omega_P`

The code uses the determinantal computational formula, not the factorial permutation sum.  For each tensor coordinate \(\alpha\), it forms the matrix \(G_\alpha=PE_\alpha\); for every pair \(j<k\), the corresponding coefficient is the signed complementary maximal minor

\[
(-1)^{j+k+1}\det G_\alpha^{\widehat{j,k}}.
\]

Thus construction is polynomial in the dense input size \(N=\prod_j r_j\).

## Numerical caveats

The current implementation is intended for exact/synthetic experiments and moderately conditioned tensors.  In particular:

- numerical ranks are determined by SVD tolerances;
- the theoretically repeated eigenvalues of `Phi` are paired numerically;
- spectral projectors use the polynomial formula from the proof, which may be ill-conditioned when two distinct spectral values are very close;
- no noisy-tensor perturbation theory is claimed here.

## Repository layout

```text
src/lp_tensor_decomposition/
    contractions.py   # Omega_P and complementary-minor matrices
    decompose.py      # Algorithm 1
    tensor_utils.py   # CP construction, compression, residuals
examples/
    basic.py
tests/
    test_contractions.py
    test_decompose.py
```

## Before public release

Add the paper citation/arXiv identifier and choose a software license before making the repository public.
