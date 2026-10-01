from multivariatepolynomial import *

import numpy as np



def test_implicit_polynomial():
    #circle
    a = 2*np.pi * np.random.rand(1000)
    x, y = np.cos(a), np.sin(a)
    X = np.column_stack([x, y])
    
    pows = [(0, 0), (2, 0), (0, 2)]
    f, p = Polynomial.fit_implicit(X, pows)
    assert f
    p = p[0].normalise((2, 0))
    assert np.allclose(p.coefs, [-1, 1, 1])
    
    
    #simple ellipse
    a = 2*np.pi * np.random.rand(1000)
    x, y = 2*np.cos(a), 3*np.sin(a)
    X = np.column_stack([x, y])
    
    pows = [(0, 0), (2, 0), (0, 2)]
    f, p = Polynomial.fit_implicit(X, pows)
    assert f
    p = p[0].normalise(0)
    assert np.allclose(p.coefs, [1, -1/2**2, -1/3**2])



def test_explicit_polynomial():
    #circle
    a = 2*np.pi * np.random.rand(1000)
    x, y = np.cos(a), np.sin(a)
    X = np.column_stack([x, y])
    
    pows = [(2, 0), (0, 2)]
    f, p = Polynomial.fit_explicit(X, np.ones_like(a), pows)
    assert f
    assert np.allclose(p.coefs, [1, 1])
    
    
    #simple ellipse
    a = 2*np.pi * np.random.rand(1000)
    x, y = 2*np.cos(a), 3*np.sin(a)
    X = np.column_stack([x, y])
    
    pows = [(2, 0), (0, 2)]
    f, p = Polynomial.fit_explicit(X, np.ones_like(a), pows)
    assert f
    assert np.allclose(p.coefs, [1/2**2, 1/3**2])
