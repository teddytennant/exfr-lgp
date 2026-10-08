# exfr_lgp

Exact finite-rank logistic Gaussian process for conditional density estimation, arXiv [2610.09452](https://arxiv.org/abs/2610.09452).

```
pip install -e .
pytest
```

The exponent is the hat-basis interpolant of equation (5). The normalizer, mean and quantiles are the closed forms in Supplement A, checked against quadrature. Nodal coefficients are updated by elliptical slice sampling under the exact likelihood.

No ADNI data, no location-varying coefficient vector, and no contraction-rate experiment. The sampler is the single-location special case, where the exponent is the nodal vector itself.
