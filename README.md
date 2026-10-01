# multivariatepolynomial

Multivariate polynomials Python package.

```python
>>> import numpy as np
>>> from multivariatepolynomial import Polynomial
>>> a = 2*np.pi * np.random.rand(1000)
>>> x, y = np.cos(a), np.sin(a)
>>> X = np.column_stack([x, y])
>>> 
>>> pows = [(0, 0), (2, 0), (0, 2)]
>>> f, (p,) = Polynomial.fit_implicit(X, pows)
>>> assert f
>>> np.allclose(p(X), 0)
True
```

## Installation

```console
pip install git+https://github.com/goessl/multivariatepolynomial.git
```

## Usage

**Enjoy the [documentation webpage](https://goessl.github.io/multivariatepolynomial).**

## Roadmap

- [x] Deploy
- [ ] Production
- [x] Ballin

## License (MIT)

Copyright (c) 2026 Sebastian Gössl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
