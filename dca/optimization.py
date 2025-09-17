"""
Optimization
------------

A class that wraps scipy.optimize.minimize, for several purposes:

    - To be able to optimize functions with signatures such as
      func(x:float, y:float, z:np.ndarray) instead of only f(x0:np.ndarray)
    - To be able to bind some function arguments and optimize over others
    - To be able to use parameters transforms, such as exp and expit,
      in order to work with bounded optimization using solves that do not
      natively support bounds

This class is strictly speaking not needed for DCA, but it encapsulates the
optimization and keeps track of a lot of bookkeeping.
"""

import inspect
import logging
import numbers
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union, Tuple

import numpy as np
import scipy as sp

from dca.utils import empty_default_params

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@dataclass(
    init=True,
    repr=True,
)
class Parameter:
    """
    An optimizable parameter with optional transformations and bounds.
    """

    x0: Union[float, np.ndarray]
    transform: Optional[Callable] = None
    transform_derivative: Optional[Callable] = None
    inverse_transform: Optional[Callable] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = (None, None)

    def __post_init__(self):
        """Data validation and setup, run after __init__."""

        assert isinstance(self.x0, (numbers.Number, np.ndarray))

        # Convert a potential integer array to float
        if isinstance(self.x0, np.ndarray):
            self.x0 = np.array(self.x0, dtype=float)

        # Initial guess x0 must be between bounds
        lb, ub = self.bounds
        if lb is not None and not np.all(self.x0 >= lb):
            log.debug(f"x0={self.x0} out of bounds [lb={lb}, ub={ub}]. Clipping.")
            self.x0 = np.maximum(lb, self.x0)
        if ub is not None and not np.all(self.x0 <= ub):
            log.debug(f"x0={self.x0} out of bounds [lb={lb}, ub={ub}]. Clipping.")
            self.x0 = np.minimum(ub, self.x0)

        # Functions used when no transforms are specified
        def identity(x):
            return x

        def one(x):
            return x * 0 + 1

        # Either all transformations must be set, or none should be set
        trans = self.transform is not None
        d_trans = self.transform_derivative is not None
        inv_trans = self.inverse_transform is not None
        params_set = [trans, d_trans, inv_trans]
        assert all(params_set) or (not any(params_set))

        # Set to identity functions if None
        if self.transform is None:
            self.transform = identity

        if self.transform_derivative is None:
            self.transform_derivative = one

        if self.inverse_transform is None:
            self.inverse_transform = identity

        # Verify properties - better to check here and avoid issues later
        for x in [-6, -3.11, 0, 3.7, 6]:
            # Check round trip mapping
            transformed = self.inverse_transform(self.transform(x))
            assert np.isclose(x, transformed), "Mapping round-trip failed"

            # Check that we map to the domain
            lb, ub = self.bounds
            if lb is not None and all(params_set):
                assert self.transform(x) >= lb
            if ub is not None and all(params_set):
                assert self.transform(x) <= ub

            # Check the derivative
            rmse = sp.optimize.check_grad(
                func=self.transform, grad=self.transform_derivative, x0=x, seed=42
            )
            assert rmse < 1e-4, "Gradient check on parameter transform failed"

    def __len__(self):
        """The length of a parameter.
        Numbers and length-1 vectors both return 1."""

        if isinstance(self.x0, numbers.Number):
            return 1
        return len(self.x0)

    def _convert_value(self, value):
        """Convert a value to the same type as the Parameter. We distinguish
        between numbers and arrays, in particular between numbers and length-1
        arrays.

        Examples
        --------
        >>> Parameter(x0=1.0)._convert_value(2.0)
        2.0
        >>> Parameter(x0=1.0)._convert_value(np.array([2]))
        2.0
        >>> Parameter(x0=np.array([1.0]))._convert_value(np.array([2.0]))
        array([2.])
        >>> Parameter(x0=np.array([1.0]))._convert_value(2.0)
        array([2.])
        """
        attr = self.x0

        if isinstance(attr, numbers.Number):
            if isinstance(value, numbers.Number):
                return value
            elif isinstance(value, np.ndarray) and len(value) == 1:
                return float(value[0])
            else:
                raise TypeError("Could not convert value")
        elif isinstance(attr, np.ndarray):
            if isinstance(value, numbers.Number):
                return np.array([value], dtype=float)
            elif isinstance(value, np.ndarray):
                return np.array(value)
            else:
                raise TypeError("Could not convert value")
        else:
            raise TypeError("Could not convert value")


class Optimizer:
    """
    Optimizes a function with support for fixed parameters, gradients,
    and parameter transforms.

    Examples
    --------

    No gradient, no transforms, but with bounds:

    >>> def loss(x):
    ...     return -np.log(x) + x
    >>> optimizer = Optimizer(loss, x=Parameter(x0=3.0, bounds=(0, None)))
    >>> opt_params, opt_result = optimizer.optimize_bfgs()
    >>> opt_params['x']
    0.9999...
    >>> opt_result.nfev
    14

    With gradients:

    >>> def grad(x):
    ...     return np.array([-1/x + 1])
    >>> optimizer = Optimizer(loss, grad=grad, x=Parameter(x0=3.0, bounds=(0, None)))
    >>> opt_params, opt_result = optimizer.optimize_bfgs()
    >>> opt_params['x']
    0.9999...
    >>> opt_result.nfev
    7
    """

    def __init__(
        self,
        loss: Callable[..., float],
        *,
        grad: Optional[Callable[..., Union[float, np.ndarray]]] = None,
        **parameters,
    ):
        self.loss = loss
        self.grad = grad
        self.parameters = dict(parameters)

        # Verify that parameters in loss and gradient are the same
        loss_params = empty_default_params(loss)
        if grad is not None:
            grad_params = empty_default_params(grad)
            assert loss_params == grad_params

        # Verify that parameters in loss and parameters send in are the same
        assert set(loss_params) == set(self.parameters.keys())

        # Sort the dictionary by the order parameters appear in function signature
        self.parameters = dict(
            sorted(
                self.parameters.items(), key=lambda tuple_: loss_params.index(tuple_[0])
            )
        )

        if self.grad is not None:
            rmse = self._check_grad()
            if rmse > 1e-1:
                warnings.warn(f"Numerical vs. analytical grad @ init {rmse=}")

    def _check_grad(self, func=None, grad=None, x0=None) -> float:
        """Returns the RMSE of the difference between analytical and numerical
        gradient."""

        if func is None:

            def func(x0):
                params = dict(self._array_to_parameters(x0))
                return self.loss(**params)

        if grad is None:

            def grad(x0):
                params = dict(self._array_to_parameters(x0))
                gradient = self.grad(**params)
                return self._filter_gradient(self.grad, gradient)

        if x0 is None:
            x0 = self._parameters_to_array(
                {n: p.x0 for (n, p) in self.parameters.items()}
            )
        return sp.optimize.check_grad(func=func, grad=grad, x0=x0, seed=42)

    def _array_to_parameters(self, array) -> dict:
        """Map an array to the parameters.

        Examples
        --------
        >>> def func(a, b, c, d):
        ...     return 1
        >>> optimizer = Optimizer(loss=func,
        ...                       a=Parameter(x0=1),
        ...                       b=Parameter(x0=np.array([2])),
        ...                       c=Parameter(x0=np.array([3, 4, 5])),
        ...                       d=Parameter(x0=6))
        >>> dict(optimizer._array_to_parameters([9, 8, 7, 6, 5, 4]))
        {'a': 9.0, 'b': array([8]), 'c': array([7, 6, 5]), 'd': 4.0}
        """
        array = np.array(array)

        # The array must match the number of parameters
        assert len(array) == sum(len(p) for p in self.parameters.values())

        idx = 0  # Start index of current slice
        for param_name, parameter in self.parameters.items():
            length = len(parameter)
            value = parameter._convert_value(array[slice(idx, idx + length)])
            yield param_name, value
            idx += length

    def _parameters_to_array(self, parameters):
        """Convert a dictionary of parameters into an array.

        Examples
        --------
        >>> def func(a, b, c, d):
        ...     return 1
        >>> optimizer = Optimizer(loss=func,
        ...                       a=Parameter(x0=1),
        ...                       b=Parameter(x0=np.array([2])),
        ...                       c=Parameter(x0=np.array([3, 4, 5])),
        ...                       d=Parameter(x0=6))
        >>> parameters = {'d': 4.0 , 'a': 9.0, 'b': np.array([8]),
        ...               'c': np.array([7, 6, 5])}
        >>> optimizer._parameters_to_array(parameters)
        array([9., 8., 7., 6., 5., 4.])
        """
        assert set(parameters.keys()) == set(self.parameters.keys())

        array = np.empty(sum(len(param) for param in self.parameters.values()))

        idx = 0  # Start index of current slice
        for param_name, parameter in self.parameters.items():
            value = parameters[param_name]
            length = len(Parameter(x0=value))
            array[slice(idx, idx + length)] = value
            idx += length

        return array

    def _filter_gradient(self, gradient_func, gradient_array):
        """The gradient function will return an array with gradients for every
        parameter, fixed or not. However, we are interested in the gradient
        with respect to free parameters only. Filter the gradient on these.

        Examples
        --------
        >>> from functools import partial
        >>> def func(a, b, c, d, k=1.0):
        ...     return np.ones(8)
        >>> func = partial(func, c=np.array([3, 4, 5]))
        >>> optimizer = Optimizer(loss=func,
        ...                       a=Parameter(x0=1),
        ...                       b=Parameter(x0=np.array([2])),
        ...                       d=Parameter(x0=np.array([6, 7, 8])))
        >>> gradient_array = np.arange(8) + 1
        >>> optimizer._filter_gradient(func, gradient_array)
        array([1, 2, 6, 7, 8])
        """
        assert isinstance(gradient_array, np.ndarray), "Gradient must return array"

        pdict = inspect.Signature.from_callable(gradient_func).parameters.items()
        free_parameters = np.zeros_like(gradient_array, dtype=bool)  # Assume fixed
        idx = 0  # Start index of current slice
        for name, parameter in pdict:
            # Empty default argument => free parameter
            if parameter.default is inspect._empty:
                length = len(self.parameters[name])
                free_parameters[slice(idx, idx + length)] = True
            else:
                length = len(Parameter(x0=parameter.default))

            idx += length
            if idx > len(gradient_array):
                break

        return gradient_array[free_parameters]

    def optimize_nelder_mead(self, **kwargs):
        """Solve the optimization problem using Nelder-Mead in the transformed
        parameter space.

        Returns a tuple (parameters:dict, result:OptimizeResult).
        """

        def fun(x0):
            # Parameters in unbounded space
            params = dict(self._array_to_parameters(x0))

            # Parameters in bounded space
            params = {n: self.parameters[n].transform(v) for (n, v) in params.items()}

            return self.loss(**params)

        # Initial guess, mapped to the unbounded space
        x0 = self._parameters_to_array(
            {n: p.inverse_transform(p.x0) for (n, p) in self.parameters.items()}
        )

        # Run optimization
        opt_result = sp.optimize.minimize(fun, x0=x0, method="Nelder-Mead", **kwargs)

        # Get parameters, then map them back to bounded space
        params = dict(self._array_to_parameters(opt_result.x))
        params = {n: self.parameters[n].transform(v) for (n, v) in params.items()}

        return params, opt_result

    def optimize_lbfgsb(self, **kwargs):
        """Solve the optimization problem using L-BFGS-B in the original
        parameter space, with bounds.

        Returns a tuple (parameters:dict, result:OptimizeResult).
        """

        if self.grad is None:
            jac = None
        else:

            def jac(x0):
                # Convert numpy array to dictionary of arguments
                params = dict(self._array_to_parameters(x0))
                # Get the gradient as an array
                gradient = self.grad(**params)
                # Only keep array indices corresponding to free params
                return self._filter_gradient(self.grad, gradient)

        def fun(x0):
            params = dict(self._array_to_parameters(x0))
            return self.loss(**params)

        # Initial guess
        x0 = self._parameters_to_array({n: p.x0 for (n, p) in self.parameters.items()})

        # Upper and lower bounds
        lb = [lb_i for p in self.parameters.values() for lb_i in len(p) * [p.bounds[0]]]
        ub = [ub_i for p in self.parameters.values() for ub_i in len(p) * [p.bounds[1]]]
        bounds = list(zip(lb, ub))

        # Run optimization
        opt_result = sp.optimize.minimize(
            fun, x0=x0, bounds=bounds, method="L-BFGS-B", jac=jac, **kwargs
        )

        return dict(self._array_to_parameters(opt_result.x)), opt_result

    def optimize_bfgs(self, check_grad=False, **kwargs):
        """Solve the optimization problem using BFGS.

        - Since BFGS does not support bounds natively, transforms are used
          to stay within bounds. Return parameters are on the bounded scale.

        Returns a tuple (parameters:dict, result:OptimizeResult).
        """

        if self.grad is None:
            jac = None
        else:

            def jac(x0):
                """Gradient. Recall the chain rule of calculus:

                unbounded-space     bounded-space      objective_value
                       x                 z                   y
                        -------t(x)-----> ---------f(z)----->
                        <------t^-1(x)---

                The derivative of the entire thing is:
                    d/dx f(t(x)) = d/dz f(z=t(x)) * d/dx t(x)

                The first factor in the product is the gradient self.grad,
                the second term is Parameter.transform_derivative.
                """
                # Convert numpy array to dictionary of arguments
                params_unbounded = dict(self._array_to_parameters(x0))

                # Parameters in bounded space
                params_bounded = {
                    n: self.parameters[n].transform(v)
                    for (n, v) in params_unbounded.items()
                }

                # Derivative of transforms, first as dict, then as a vector
                transforms_derivatives = {
                    n: self.parameters[n].transform_derivative(v)
                    for (n, v) in params_unbounded.items()
                }
                gradient_transforms = self._parameters_to_array(transforms_derivatives)

                # Get the gradient as an array
                gradient = self.grad(**params_bounded)
                # Only keep array indices corresponding to free params, apply chain rule
                return self._filter_gradient(self.grad, gradient) * gradient_transforms

        def fun(x0):
            # Parameters in unbounded space
            params = dict(self._array_to_parameters(x0))

            # Parameters in bounded space
            params = {n: self.parameters[n].transform(v) for (n, v) in params.items()}

            return self.loss(**params)

        # Initial guess, mapped to the unbounded space
        x0 = self._parameters_to_array(
            {n: p.inverse_transform(p.x0) for (n, p) in self.parameters.items()}
        )

        # Check gradient (to verify that chain rule works)
        if jac is not None and check_grad:
            grad_rmse = self._check_grad(func=fun, grad=jac, x0=x0)
            if grad_rmse > 1e-1:
                warnings.warn(f"Grad check failed on chain rule {x0=}. {grad_rmse=}")

        # Run optimization
        opt_result = sp.optimize.minimize(fun, x0=x0, method="BFGS", jac=jac, **kwargs)

        # Get parameters, then map them back to bounded space
        params = dict(self._array_to_parameters(opt_result.x))
        params = {n: self.parameters[n].transform(v) for (n, v) in params.items()}

        return params, opt_result

    def optimize_tnc(self, **kwargs):
        """Solve the optimization problem using  truncated Newton (TNC)
        algorithm in the original parameter space, with bounds.

        Returns a tuple (parameters:dict, result:OptimizeResult).
        """

        if self.grad is None:
            jac = None
        else:

            def jac(x0):
                # Convert numpy array to dictionary of arguments
                params = dict(self._array_to_parameters(x0))
                # Get the gradient as an array
                gradient = self.grad(**params)
                # Only keep array indices corresponding to free params
                return self._filter_gradient(self.grad, gradient)

        def fun(x0):
            params = dict(self._array_to_parameters(x0))
            return self.loss(**params)

        # Initial guess
        x0 = self._parameters_to_array({n: p.x0 for (n, p) in self.parameters.items()})

        # Upper and lower bounds
        lb = [lb_i for p in self.parameters.values() for lb_i in len(p) * [p.bounds[0]]]
        ub = [ub_i for p in self.parameters.values() for ub_i in len(p) * [p.bounds[1]]]
        bounds = list(zip(lb, ub))

        # Run optimization
        opt_result = sp.optimize.minimize(
            fun, x0=x0, bounds=bounds, method="TNC", jac=jac, **kwargs
        )

        return dict(self._array_to_parameters(opt_result.x)), opt_result


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys", "-x"])
