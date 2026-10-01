from multivariatepolynomial import *
import numpy as np
from sklearn.preprocessing import PolynomialFeatures



def sklearn_powers(degree, n_features):
    return PolynomialFeatures(degree).fit(np.zeros((1, n_features))).powers_

def test_powers_total_degree():
    for n_features in range(1, 6):
        for degree in range(0, 6):
            P = powers(degree, n_features=n_features)
            assert P.dtype == np.uint8
            assert np.array_equal(P, sklearn_powers(degree, n_features))

def test_powers_per_feature():
    degrees = [
        (0,), (3,), (0, 0), (2, 1), (1, 2), (0, 3), (3, 3),
        (2, 0, 1), (1, 1, 1), (3, 2, 1, 0), (1, 2, 1, 2, 1)
    ]
    for d in degrees:
        P = sklearn_powers(sum(d), len(d))
        expected = P[np.all(P<=d, axis=1)]
        
        P = powers(*d)
        assert P.dtype == np.uint8
        assert np.array_equal(P, expected)



def test_transform():
    for _ in range(10):
        #PolynomialFeatures requires at least one sample and one feature
        #and can't handle degree=0 while include_bias=False
        n_samples, n_features = np.random.randint(1, 10, size=2)
        degree = np.random.randint(0, 4)
        include_bias = bool(np.random.randint(0, 2)) or not bool(degree)
        
        X = np.random.rand(n_samples, n_features)
        features = PolynomialFeatures(degree, include_bias=include_bias).fit(X)
        pows = features.powers_
        
        assert np.allclose(transform(X, pows), features.transform(X))
        
        #special case: empty samples
        empty_samples = np.empty((0, n_features))
        assert transform(empty_samples, pows).shape == (0, pows.shape[0])
        
        #special case: single sample x vector
        assert np.allclose(transform(X[0], pows), features.transform(X[:1])[0])
