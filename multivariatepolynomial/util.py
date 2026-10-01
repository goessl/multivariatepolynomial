"""Multivariate polynomial helpers."""



from itertools import combinations_with_replacement, product
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Any



__all__ = ('powers', 'transform', )



def _rank(S: NDArray, shape: tuple[int, ...]) -> int:
    """Return the rank from singular values.
    
    Zero tolerance acc. to [`numpy.linalg.matrix_rank`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html),
    but also for empty matrices.
    """
    tol = S.max(initial=0) * max(shape, default=0) * np.finfo(S.dtype).eps
    return int(np.count_nonzero(S > tol))

def _safe_div(num: ArrayLike, den: ArrayLike) -> NDArray:
    """Return `num/den`, `0` where `den==0`."""
    num, den = np.asarray(num), np.asarray(den)
    r = np.zeros(np.broadcast_shapes(num.shape, den.shape),
                 dtype=np.result_type(num, den, 0.0))
    return np.divide(num, den, out=r, where=den!=0)


def powers(*degrees: int, n_features: int|None=None) -> NDArray[np.uint8]:
    r"""Return an exponent array.
    
    Similar to `sklearn.preprocessing.PolynomialFeatures.powers_`.
    
    Limited to `uint8`.
    
    Parameters
    ----------
    degrees
        Maximum total degree, or maximum degree of each feature.
    n_features
        Number of features, if only the total degree is given.
    
    Returns
    -------
    :
        Exponents in graded lexicographic order (grlex).
    """
    if not len(degrees) >= 1:
        raise TypeError('requiring at least one degree value')
    if not all(isinstance(d, int) for d in degrees):
        raise TypeError('degrees must be integers')
    if not all(0<=d for d in degrees):
        raise ValueError('degrees must be non-negative')
    
    if not (isinstance(n_features, int) or n_features is None):
        raise TypeError('n_features must be an integer or None')
    
    if isinstance(n_features, int):
        if not n_features>=0:
            raise ValueError('n_features must be non-negative')
        if len(degrees)!=1:
            raise TypeError('either varargs degrees, or one degree and n_features')
        
        d = degrees[0]
        #pows = [p for p in product(range(d+1), repeat=n_features) if sum(p)<=d]
        #pows = np.array(pows, dtype=np.uint8)
        #faster:
        pows = [np.bincount(c, minlength=n_features)
                for k in range(d+1)
                for c in combinations_with_replacement(range(n_features), k)]
        pows = np.array(pows, dtype=np.uint8).reshape(len(pows), n_features)
    else:
        pows = list(product(*(range(d+1) for d in degrees)))
        pows = np.array(pows, dtype=np.uint8)
    
    #grlex
    pows = pows[np.lexsort((*(np.iinfo(pows.dtype).max-pows.T[::-1]), pows.sum(axis=1)))]
    
    return pows


def transform(X: ArrayLike, pows: ArrayLike) -> NDArray[Any]:
    r"""Polynomial transform features to monomials.
    
    $$
        \begin{pmatrix}
            \vec{x}_i^{\alpha_j}
        \end{pmatrix}_{ij}
    $$
    
    Basically `sklearn.preprocessing.PolynomialFeatures.transform`
    but with custom exponents and type agnostic.
    
    Parameters
    ----------
    X
        Feature matrix or vector. Each row becomes a row in the result.
    pows
        Exponents. Each row becomes a column in the result.
    
    Returns
    -------
    :
        Monomials.
    """
    X, pows = np.asarray(X), np.asarray(pows)
    if X.ndim not in {1, 2}:
        raise ValueError('X must be one or two dimensional')
    if pows.ndim != 2:
        raise ValueError('pows must be two dimensional')
    if X.shape[-1] != pows.shape[1]:
        raise ValueError('X must have as many features as pows (last axis)')
    
    #return np.prod(X[:, np.newaxis, :] ** pows, axis=2)
    #more memory efficient:
    r = np.empty(X.shape[:-1]+(pows.shape[0],), dtype=np.result_type(X, pows))
    for i, p in enumerate(pows):
        r[...,i] = np.prod(X**p, axis=-1)
    return r
