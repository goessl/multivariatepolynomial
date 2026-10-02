"""Multivariate polynomials."""



from __future__ import annotations
from warnings import warn
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sympy import Symbol, symbols, Poly
from .util import _rank, _safe_div, transform
from typing import Any, Final, Self
from collections.abc import Callable, Iterator



__all__ = ('Polynomial', )



class Polynomial:
    r"""Multivariate polynomial.
    
    $$
        p(x_0, x_1, \dots, x_{\text{n_features}-1})
    $$
    """
    coefs: Final[NDArray]
    """Coefficients. Shape `(n_terms,)`."""
    pows: Final[NDArray]
    """Monomial exponents. Shape `(n_terms, n_features)`."""
    
    
    @classmethod
    def fit_implicit(cls, X: ArrayLike, pows: ArrayLike, *,
            nullspace: Callable[[NDArray], NDArray]|None=None) \
            -> tuple[bool, tuple[Self, ...]]:
        """Return fitted implicit `Polynomial`s.
        
        Flag is `True` if a single polynomial converged.
        `False` if either none or more than one converged.
        
        If none converged, the best candidate is still returned
        (without custom `nullspace` argument).
        If one or more converged, all converged are returned.
        
        Parameters
        ----------
        X
            Samples. Two dimensional `(n_samples, n_features)`.
        pows
            Monomial exponents. Two dimensional `(n_terms, n_features)`.
        nullspace
            Optional custom nullspace calculating function.
            Should accept the design matrix and return a kernel basis as columns.
        
        Returns
        -------
        :
            Flag if unique fit converged and implicit polynomials.
        """
        X, pows = np.asarray(X), np.asarray(pows)
        if X.ndim != 2:
            raise ValueError('X must be two dimensional')
        if pows.ndim != 2:
            raise ValueError('pows must be two dimensional')
        if X.shape[-1] != pows.shape[1]:
            raise ValueError('X must have as many features as pows (last axis)')
        
        if X.shape[0] < pows.shape[0]:
            warn(f'fewer samples than coefficients to fit (coverage: {X.shape[0]/pows.shape[0]})', UserWarning, stacklevel=2)
        
        A = transform(X, pows)
        
        if nullspace is None:
            #keep singular values for diagnostics
            _, S, Vt = np.linalg.svd(A, full_matrices=A.shape[0]<A.shape[1])
            K = Vt.conj().T[:,::-1] #vectors as columns, starting with best
            k = pows.shape[0] - _rank(S, A.shape)
        else:
            K = nullspace(A)
            k = K.shape[1]
        
        if k==0 and K.shape[1]==0: #no solution at all
            warn('No solution found!', UserWarning, stacklevel=2)
            return False, ()
        elif k==0 and K.shape[1]>0: #no exact solution
            warn('Fit didn\'t converge!', UserWarning, stacklevel=2)
            return False, (cls(K[:,0], pows),) #still return best candidate
        elif k > 1: #multiple exact solutions
            warn(f'Kernel dimensionality {k}; solution is not unique!',
                    UserWarning, stacklevel=2)
            return False, tuple(cls(c, pows) for c in K[:,:k].T)
        #one exact solution
        return True, (cls(K[:,0], pows),)
    
    @classmethod
    def fit_explicit(cls, X: ArrayLike, y: ArrayLike, pows: ArrayLike) \
            -> tuple[bool, Self]:
        r"""Return the best fitted explicit `Polynomial`.
        
        $$
            y = p(\vec{x})
        $$
        
        Flag is `True` if a single polynomial converged.
        `False` if either none or more than one converged.
        
        If none converged, the least squares solution is still returned.
        If more than one converged, they differ by the implicit polynomials of
        [`fit_implicit(X, pows)`][multivariatepolynomial.polynomial.Polynomial.fit_implicit].
        
        Parameters
        ----------
        X
            Samples. Two dimensional `(n_samples, n_features)`.
        y
            Targets. One dimensional `(n_samples,)`.
        pows
            Monomial exponents. Two dimensional `(n_terms, n_features)`.
        
        Returns
        -------
        :
            Flag if unique fit converged and explicit polynomial.
        """
        X, y, pows = np.asarray(X), np.asarray(y), np.asarray(pows)
        if X.ndim != 2:
            raise ValueError('X must be two dimensional')
        if y.ndim != 1:
            raise ValueError('y must be one dimensional')
        if pows.ndim != 2:
            raise ValueError('pows must be two dimensional')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X must have as many samples as y')
        if X.shape[-1] != pows.shape[1]:
            raise ValueError('X must have as many features as pows (last axis)')
        
        if X.shape[0] < pows.shape[0]:
            warn(f'fewer samples than coefficients to fit (coverage: {X.shape[0]/pows.shape[0]})', UserWarning, stacklevel=2)
        
        A = transform(X, pows)
        
        #minimum norm least squares
        c, _, r, S = np.linalg.lstsq(A, y)
        k = pows.shape[0] - r
        #don't package polynomial and calculate residual with it
        #because it would calculate the transformation matrix again
        res = np.linalg.norm(A@c - y)
        tol = max(A.shape[0], A.shape[1]+1) * np.finfo(S.dtype).eps \
                * max(S.max(initial=0), np.linalg.norm(y))
        
        if res > tol: #no exact solution
            warn(f'Fit didn\'t converge: {res}!', UserWarning, stacklevel=2)
        if k > 0: #multiple solutions
            warn(f'Kernel dimensionality {k}; solution is not unique!',
                    UserWarning, stacklevel=2)
        return res<=tol and k==0, cls(c, pows)
    
    
    def __init__(self, coefs: ArrayLike, pows: ArrayLike) -> None:
        coefs, pows = np.array(coefs), np.array(pows)
        if coefs.ndim != 1:
            raise ValueError('coefs must be one dimensional')
        if pows.ndim != 2:
            raise ValueError('pows must be two dimensional')
        if coefs.shape[0] != pows.shape[0]:
            raise ValueError('there must be as many coefficients as exponents vectors')
        
        self.coefs, self.pows = coefs, pows
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.pows.shape[1]
    
    def __len__(self) -> int:
        """Return the number of terms.
        
        Returns
        -------
        :
            Number of terms.
        """
        return self.pows.shape[0]
    
    def trim(self) -> Polynomial:
        """Return a trimmed copy.
        
        Removes all terms where `numpy.isclose(c, 0)`.
        
        Returns
        -------
        :
            Trimmed copy.
        """
        i = ~np.isclose(self.coefs, 0)
        return type(self)(self.coefs[i], self.pows[i,:])
    
    def normalise(self, c_or_i: int|tuple[int,...]) -> Polynomial:
        """Return a normalised copy by the specified coefficient.
        
        Parameters
        ----------
        c_or_i
            Either unraveled coefficient index
            or exponent tuple of the corresponding monomial.
        
        Returns
        -------
        :
            Normalised copy.
        """
        if isinstance(c_or_i, int): #normalise by unraveled index
            return type(self)(self.coefs/self.coefs[c_or_i], self.pows)
        elif isinstance(c_or_i, tuple): #normalise by exponent index
            for i, c in enumerate(self.pows):
                if c_or_i == tuple(map(int, c)):
                    return type(self)(self.coefs/self.coefs[i], self.pows)
            raise ValueError('exponent not found')
        raise TypeError('c_or_i must be an integer or tuple of integers')
    
    def __bool__(self) -> bool:
        """Return whether any coefficient is non-zero.
        
        Tries using `numpy.isclose(c, 0)`, falls back to `bool(c)`.
        
        Returns
        -------
        :
            Whether any coefficient is non-zero.
        """
        for v in self.coefs:
            try:
                if not np.isclose(v, 0):
                    return True
            except TypeError:
                if bool(v):
                    return True
        return False
    
    def __iter__(self) -> Iterator[tuple[tuple[int,...], Any]]:
        """Yield the terms like a `dict`.
        
        Yields
        ------
        :
            Tuples of the exponents, mapped to integer tuples
            with their corresponding coefficients.
        """
        for k, v in zip(self.pows, self.coefs, strict=True):
            yield tuple(map(int, k)), v
    
    def __call__(self, X: ArrayLike) -> NDArray:
        r"""Return the evaluation.
        
        $$
            p(X)
        $$
        
        Parameters
        ----------
        X
            Input vector or matrix $X$.
        
        Returns
        -------
        :
            This polynomial evaluated at $X$.
        """
        X = np.asarray(X)
        if X.ndim not in {1, 2}:
            raise ValueError('X must be one or two dimensional')
        if X.shape[-1] != self.n_features:
            raise ValueError(
                    'X must have as many features as this polynomial (last axis)')
        
        return transform(X, self.pows) @ self.coefs
    
    def relative_residual(self, X: ArrayLike) -> NDArray:
        r"""Return the relative residual.
        
        $$
            \frac{|\sum_\alpha c_\alpha\vec{x}^\alpha|}{\sum_\alpha|c_\alpha\vec{x}^\alpha|}
        $$
        
        Parameters
        ----------
        X
            Input vector or matrix $X$.
        
        Returns
        -------
        :
            This relative residual.
        """
        X = np.asarray(X)
        if X.ndim not in {1, 2}:
            raise ValueError('X must be one or two dimensional')
        if X.shape[-1] != self.n_features:
            raise ValueError(
                    'X must have as many features as this polynomial (last axis)')
        
        A = transform(X, self.pows)
        return _safe_div(
            np.abs(A@self.coefs),
            np.sum(np.abs(A*self.coefs), axis=-1)
        )
    
    def to_sympy(self, *gens: Symbol, trim: bool=True) -> Poly:
        """Return as a `sympy.Poly`.
        
        Parameters
        ----------
        gens
            Generators.
        trim
            Optional trim flag. Will remove near zero coefficients.
        
        Returns
        -------
        :
            Sympified polynomial.
        """
        if not gens:
            gens = symbols(f'x:{self.n_features}')
        else:
            if len(gens) != self.n_features:
                raise ValueError('must provide none or n_feature many generators')
        
        return Poly.from_dict({
            k: v for k, v in self if not trim or not np.isclose(v, 0)
        }, *gens)
    
    def _sympy_(self) -> Poly:
        return self.to_sympy()
