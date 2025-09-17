"""
Utility functions for DCA.
"""

import inspect
from typing import Callable

import numpy as np
import scipy as sp


def empty_default_params(function: Callable) -> list:
    """Given a function, return a list of function parameters with no defaults.
    The parameters are ordered by the order that they appear in the signature.

    Examples
    --------
    >>> def func(a, b, c, d=1, e=None):
    ...     return 1
    >>> empty_default_params(func)
    ['a', 'b', 'c']
    """
    return [
        param_name
        for (param_name, parameter) in inspect.Signature.from_callable(
            function
        ).parameters.items()
        if parameter.default is inspect._empty
    ]


def weighted_linregress(x, y, *, w=None):
    """Solve weighted linear least squares, returning (slope, intercept).

    Examples
    --------
    >>> x = np.array([1, 1, 1, 2, 2, 3])
    >>> y = np.array([0.2, 0.2, 0.2, 0.5, 0.5, 0.4])
    >>> slope, intercept = weighted_linregress(x, y)

    Verify that the results match sp.stats.linregress:

    >>> from scipy.stats import linregress
    >>> results = linregress(x, y)
    >>> close_slope = bool(np.isclose(results.slope, slope))
    >>> close_intercept = bool(np.isclose(results.intercept, intercept))
    >>> close_slope and close_intercept
    True

    Verify the weight property (integer weights equal repeating data):

    >>> slope, _ = weighted_linregress(x, y)
    >>> slope
    0.14000...
    >>> slope, _ = weighted_linregress([1, 2, 3], [0.2, 0.5, 0.4], w=[3, 2, 1])
    >>> slope
    0.14000...
    """
    w = np.ones_like(x, dtype=float) if w is None else np.array(w, dtype=float)

    x, y = np.array(x, dtype=float), np.array(y, dtype=float)

    assert len(x) == len(y) == len(w)

    # Create equations
    A = np.vstack([x, np.ones(len(x))]).T
    AW = A * w[:, None]
    YW = y * w

    # Solving the normal equation, see:
    # https://en.wikipedia.org/wiki/Weighted_least_squares#Formulation
    lhs = AW.T @ A
    EPSILON = np.sqrt(np.finfo(float).eps)  # 1.49e-08
    np.fill_diagonal(lhs, lhs.diagonal() + EPSILON)  # Regularize
    rhs = A.T @ YW
    params = sp.linalg.solve(lhs, rhs, assume_a="pos")

    return tuple(float(param) for param in params)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
