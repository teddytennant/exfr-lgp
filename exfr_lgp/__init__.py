"""Exact finite-rank logistic Gaussian process, arXiv 2610.09452.

The conditional log density is g(y, w) - log Z, with g piecewise linear on a
uniform grid (equation 5, hat basis 6). Z, the mean and the quantiles are the
closed forms in Supplement A.1, A.2 and A.3. The Gaussian nodal coefficients
are updated by elliptical slice sampling (Murray, Adams and MacKay, 2010).
"""

from exfr_lgp.density import (
    hat,
    interpolate,
    log_density,
    log_normalizer,
    mean,
    nodal_values,
    quantile,
    sample_density,
)
from exfr_lgp.ess import elliptical_slice, fit_nodes

__all__ = [
    "hat",
    "interpolate",
    "log_density",
    "log_normalizer",
    "mean",
    "nodal_values",
    "quantile",
    "sample_density",
    "elliptical_slice",
    "fit_nodes",
]
