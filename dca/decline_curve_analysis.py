"""
Curve functions
---------------

This module contains curves like Arps. These decline curves can be evaluated,
transformed in various ways, integrated, differentiated with respect to the
parameters, and so on.

>>> arps = Arps(10, 5, -1)
>>> arps
Arps(10.0, 5.0, -1.0)
>>> arps.original_parametrization()
(148.41..., 0.26..., 0.0092...)
>>> float(arps.eval_integral_from_0_to_inf())
22026.46...

>>> time = np.linspace(0, 10, num=4)
>>> arps.eval_log(time)
array([5.        , 4.96... , 4.93... , 4.90...])
>>> arps.eval_grad_log(time).sum(axis=1)
array([ 4.        , -3.81..., -0.047...])
"""

import dataclasses
import logging
import numbers
import operator

import numpy as np
from scipy.special import expit as logistic_sigmoid

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class CurveLoss:
    r"""Create a curve loss instance.

    Parameters
    ----------
    curve_func : callable
        A callable with signature (x, theta1, theta2, theta3), returning y.
        The thetas will be optimized by curve fitting.
    mu : np.ndarray, optional
        A vector of prior means for curve fitting parameters.
        The default is None (interpreted as no prior).
    precision_matrix : np.ndarray, optional
        A matrix of prior precisions for curve fitting parameters.
        (The inverse of the covariance matrix).
        The default is None (interpreted as no prior).
    p : float, optional
        The p in the p-norm used in the loss function. The default is 1.4.
    half_life : float, optional
        Half life used to create weights for exponential decay, weighting
        the most recent data points more. For instance, if half_life=100,
        then a data point today is worth twice as much as a data point
        from 100 days ago, and four times as much as a data point 200 days
        ago. The default is None (interpreted as infinity, or no decay).
    curve_func_grad : callable
        A callable with signature (x, theta1, theta2, theta3), which returns
        a numpy array of shape (3, len(x)) representing the gradient at each
        point in x.

    Notes
    -----

    Loss function used for curve fitting.

    We use a class to set up common hyperparmeters.

    Curves are fit using a multivariate Gaussian prior. An instance of this
    class is meant to be used with `scipy.optimize.minimize`.

    The loss function is:

    .. math::

            \mathcal{L}(\boldsymbol{\theta}) =
            \sum_i w_i (\gamma) \lvert \ln y_i - \ln f(t_i; \boldsymbol{\theta}) \rvert ^p
            + \alpha \left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right)^T
            \boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1}
            \left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right)

    where,

        * :math:`w_i(\gamma)` is a weighting function that scales up the contribution of the most recent data,
        * the choice of whether we use a logarithm or not is determined by cross-validation,
        * :math:`1 \leq p \leq 2` is a hyperparameter determined by cross-validation,
        * a multivariate normal prior is put on :math:`\boldsymbol{\theta}` (the second term with the quadratic function)
        * the strength of the prior is given by :math:`\alpha` and determined by cross-validation.


    Regularization of the prior is obatined by scaling the precision_matrix
    with a scalar value.


    Examples
    --------
        >>> import numpy as np
        >>> from scipy.optimize import minimize
        >>> mu = np.array([14.615444, 1.879282, 0.004616])
        >>> precision_matrix = np.array([[ 0.80, -0.32, -0.11],
        ...                              [-0.32,  0.67, -0.19],
        ...                              [-0.11, -0.19,  0.26]])
        >>> def log_Arps(t, theta1, theta2, theta3):
        ...     return Arps(theta1, theta2, theta3).eval_log(t)
        >>> loss_function = CurveLoss(mu=mu, precision_matrix=precision_matrix,
        ...                           curve_func=log_Arps)
        >>> x = np.arange(5)
        >>> y = np.array([10, 8, 6, 4, 1])
        >>> optimization_result = minimize(loss_function, x0=np.array([1, 1, 1]),
        ...                                method="BFGS", args=(x, y,),)
        >>> np.allclose(optimization_result.x,
        ...             np.array([14.38081254, 4.30188387, 7.62215325]))
        True

    Returns
    -------

        None.
    """

    def __init__(
        self,
        *,
        curve_func,
        mu=None,
        precision_matrix=None,
        p=1.4,
        half_life=None,
        curve_func_grad=None,
    ):
        # Copy arguments if they are not None
        self.mu = mu if mu is None else np.array(mu)
        self.precision_matrix = (
            precision_matrix if precision_matrix is None else np.array(precision_matrix)
        )
        self.curve_func = curve_func
        self.p = p
        self.half_life = half_life
        self.curve_func_grad = curve_func_grad

        assert (half_life is None) or (
            isinstance(half_life, numbers.Real) and half_life > 0
        ), "`half_life` must be a positive, real number"

        assert self.p >= 1 and self.p <= 2, "`p` must be in [1, 2]"
        assert callable(self.curve_func), "`curve_func` must be callable"
        assert callable(self.curve_func_grad) or curve_func_grad is None, (
            "`curve_func_grad` must be callable or None"
        )

        assert not operator.xor(self.mu is None, self.precision_matrix is None), (
            "Both `mu` and `precision_matrix` must either be None, or given."
        )

        # Both were given, check that they match up
        if self.mu is not None and self.precision_matrix is not None:
            assert self.mu.ndim == 1, "`mu` must be a vector"
            assert self.precision_matrix.ndim == 2, (
                "`precision_matrix` must be a matrix"
            )
            assert self.precision_matrix.shape[0] == self.precision_matrix.shape[1], (
                "`precision_matrix` must be square"
            )
            assert self.mu.size * self.mu.size == self.precision_matrix.size, (
                "`mu` and `precision_matrix` must have the same number of dimensions"
            )

    @staticmethod
    def _weight(w, half_life=None):
        """Compute the weights using the half life.

        Examples
        --------
        >>> w = np.array([1, 1, 1, 1, 1, 1])
        >>> CurveLoss._weight(w, half_life=2)
        array([0.3550402 , 0.50210266, 0.71008039, 1.00420532, 1.42016079,
               2.00841064])
        >>> float(CurveLoss._weight(w, half_life=2).sum())
        6.0

        Or with a different half-life, e.g. 1:

        >>> CurveLoss._weight(w, half_life=1)
        array([0.0952381 , 0.19047619, 0.38095238, 0.76190476, 1.52380952,
               3.04761905])

        """
        assert np.all(w >= 0) and np.all(w <= 1)

        # Create half-life weights
        if half_life is None:
            weights_hl = np.ones_like(w) / len(w) * np.sum(w)
        else:
            # Create weights based on exponential time-decay, then normalize them
            # For instance, if w = [0.25, 0.5, 1.0, 1.0]
            cumsum_reversed = np.cumsum(w[::-1])[::-1]
            # cumsum_reversed = [2.75, 2.5, 2, 1]
            weights_hl = 2.0 ** (-cumsum_reversed / half_life)
            # weights_hl == [0.3855, 0.4204, 0.5, 0.7071]
            weights_hl = weights_hl / np.sum(weights_hl) * np.sum(w)  # Sum to N
            # weights_hl == [0.5266, 0.5743, 0.6830, 0.9659]

        assert np.isclose(np.sum(weights_hl), np.sum(w)), "Weights must sum to sum(w)"

        return weights_hl

    def __call__(self, thetas, x, y, weights=None):
        """Get the loss of the curve.

        Parameters
        ----------
        thetas : np.ndarray
            Parameters to be optimized.
        x : np.ndarray
            The x-values (typically time).
        y : np.ndarray
            The y-values.
        weights : np.ndarray
            Weights used for each data point. Will be multiplied with existing
            weights given by the half life parameter, if they exist.

        Returns
        -------
        float
            Negative log posterior, proportional to log(likelihood) + log(prior).
            Minimizing this is the same as maximizing the likelihood.

        """
        x = np.asarray_chkfinite(x)
        y = np.asarray_chkfinite(y)
        weights = np.ones_like(x) if weights is None else np.asarray_chkfinite(weights)
        assert x.shape == y.shape == weights.shape
        assert x.ndim == 1

        # No observations, do not use likelihood at all
        if len(x) == 0:
            assert self.mu is not None and self.precision_matrix is not None
            return (thetas - self.mu).T @ self.precision_matrix @ (thetas - self.mu)

        # Create half-life weights
        weights_hl = self._weight(weights, half_life=self.half_life)

        # Create data weights by multiplying user-weights with time-decay weights
        weights = weights * weights_hl

        # Construct negative log-likelihood, the first term in the loss function
        if self.p is not None:
            log_likelihood = np.sum(
                weights * np.abs(self.curve_func(x, *thetas) - y) ** self.p
            )
        else:
            raise ValueError("Variable `p` must be set.")

        if self.mu is not None and self.precision_matrix is not None:
            # The quadratic form is the negative log-prior used to regularize the loss
            log_prior = (
                (thetas - self.mu).T @ self.precision_matrix @ (thetas - self.mu)
            )

        else:
            log_prior = 0  # No prior

        # log.debug(f"CurveLoss at theta {thetas} -> {log_likelihood + log_prior}")
        return log_likelihood + log_prior

    def grad(self, thetas, x, y, weights=None):
        """Gradient of the loss function.

        Returns an array of shape (num_parameters,). For instance, if the DCA
        curve is the Arps curve, then a length-3 array is returned, which
        represents the objective function differentiated with respect to
        theta_1, theta_2 and theta_3.
        """
        x = np.asarray_chkfinite(x)
        y = np.asarray_chkfinite(y)
        weights = np.ones_like(x) if weights is None else np.asarray_chkfinite(weights)
        assert x.shape == y.shape == weights.shape
        assert x.ndim == 1

        # No observations, do not use likelihood at all
        if len(x) == 0:
            assert self.mu is not None and self.precision_matrix is not None
            return 2 * self.precision_matrix @ (thetas - self.mu)  # Gradient

        # Create half-life weights
        weights_hl = self._weight(weights, half_life=self.half_life)

        # Create data weights by multiplying user-weights with time-decay weights
        weights = weights * weights_hl

        # Construct negative log-likelihood, the first term in the loss function
        if self.p is not None:
            # Compute the gradient of |f(x, theta) - y|^p
            residuals = self.curve_func(x, *thetas) - y

            # Has shape (len(x),)
            grad_residual = (
                self.p * np.abs(residuals) ** (self.p - 1) * np.sign(residuals)
            )

            # Has shape (len(thetas), len(x))
            grad_f = self.curve_func_grad(x, *thetas)

            # Compute the gradient over all points in the sum, then sum it
            grad = (grad_f * grad_residual * weights).sum(axis=1)
            assert grad.shape == (len(thetas),)

        else:
            raise ValueError("Variable `p` must be set.")

        if self.mu is not None and self.precision_matrix is not None:
            # The quadratic form is the negative log-prior used to regularize the loss
            grad_prior = 2 * self.precision_matrix @ (thetas - self.mu)

        else:
            grad_prior = 0  # No prior

        assert (grad + grad_prior).shape == (len(thetas),)
        # log.debug(f"Gradient of CurveLoss at theta {thetas} -> {grad + grad_prior}")
        return grad + grad_prior


@dataclasses.dataclass
class Arps:
    r"""Create an Arps instance.

    Parameters
    ----------
    theta1 : float
        Parameter number one. Interpreted as the logarithm of the integral from
        zero to infinity. This scales the generalized Pareto distribution.
    theta2 : float
        Parameter number two.
    theta3 : float
        Parameter number three.

    Notes
    -----

    We use the parametrization given in the paper 'Bayesian Hierarchical Modeling'
    by Se Yoon Lee et al by default. This parametrization is over unbounded
    variables :math:`\boldsymbol{\theta} = (\theta_1, \theta_2, \theta_3)`.

    .. math::

        \mu(t; \boldsymbol{\theta}) =
        \exp \left(
            \theta_1 - \theta_2 - (1 + \exp(-\theta_3))
            \log(1 + t \exp(\theta_3 - \theta_2))
            \right)

    It's also possible to use the "original" parametrization with parameters
    :math:`(q_1, h, D)`

    See also:
        https://en.wikipedia.org/wiki/Generalized_Pareto_distribution

    Examples
    --------
    >>> model = Arps(1, -2, 1)
    >>> time = np.arange(5)
    >>> model(time)
    array([20.08...,  0.31...,  0.12...,  0.072... ,  0.04...])

    >>> model = Arps.from_original_parametrization(q_1=10, h=0.66, D=2)

    """

    def __init__(self, theta1: float, theta2: float, theta3: float):
        # See the paper 'Bayesian Hierarchical Modeling' by Se Yoon Lee et al
        # This class implements equation (3.3) in the paper, and integrals
        self.theta1 = float(theta1)
        self.theta2 = float(theta2)
        self.theta3 = float(theta3)

    def __repr__(self):
        return f"Arps{self.thetas}"

    @property
    def thetas(self):
        """Return the thetas as a tuple."""
        return (self.theta1, self.theta2, self.theta3)

    def eval_log(self, t):
        r"""Evaluate :math:`\log \mu(t; \boldsymbol{\theta})`."""
        t = np.array(t)
        assert np.all(np.isfinite(t)), "All values must be finite and not NaN"

        t1 = self.theta1
        t2 = self.theta2
        t3 = self.theta3

        logterm = log1pexp(t=t, x=t3 - t2)

        exp_t3 = np.exp(-t3)

        # Equivalent to t1 - t2 - (1 + np.exp(-t3)) * np.log1p(t * np.exp(t3 - t2))
        return t1 - t2 - (1 + exp_t3) * logterm

    def eval(self, t):
        r"""Evaluate :math:`\mu(t; \boldsymbol{\theta})`."""
        return np.exp(self.eval_log(t))

    def __call__(self, t):
        return self.eval(t)

    def add(self, k):
        """Add a constant k in log space and return a new instance.

        Examples
        --------
        >>> model = Arps(10, 3, 1)
        >>> model.add(2)
        Arps(12.0, 3.0, 1.0)
        """
        t1 = self.theta1 + k
        t2 = self.theta2
        t3 = self.theta3
        return type(self)(t1, t2, t3)

    def shift_and_scale(self, shift=0, scale=1):
        r"""Scale and shift the curve along the horizontal axis.

        Equivalent to fitting on a transformed grid:
            :math:`t' = t \times \text{scale} + \text{shift}`.

        Examples
        --------
        >>> curve = Arps(10, 5, 1)
        >>> x_grid = np.array([0, 1, 2, 3, 5])
        >>> curve.eval_log(x_grid)
        array([5.        , 4.97517309, 4.95078876, 4.92683151, 4.8801402 ])
        >>> curve.shift_and_scale(scale=2, shift=3).eval_log((x_grid * 2 + 3))
        array([5.        , 4.97517309, 4.95078876, 4.92683151, 4.8801402 ])
        """
        assert isinstance(shift, numbers.Real)
        assert isinstance(scale, numbers.Real)
        assert scale > 0, "Scale must be positive"
        # Transform to original parametrization space
        q_1, h, D = self.original_parametrization()

        # Transform the arps equation
        # q_new = q_1 * (1 - shift * h * D / scale) ** (-1 / h)
        # Increase numerical stability:
        q_new = q_1 * np.exp((-1 / h) * np.log1p(-shift * h * D / scale))
        D_new = 1 / ((scale / D) - (h * shift))

        # Return a new instance based on new parameters
        return self.from_original_parametrization(q_1=q_new, h=h, D=D_new)

    def eval_integral(self, t):
        r"""Evaluate the anti-derivative of :math:`\mu(t; \boldsymbol{\theta})`."""
        t1 = self.theta1
        t2 = self.theta2
        t3 = self.theta3
        logterm = log1pexp(t=t, x=t3 - t2)
        return -np.exp(t1 - np.exp(-t3) * logterm)

    def eval_squared_integral(self, t):
        r"""Evaluate the anti-derivative of :math:`\mu(t; \boldsymbol{\theta})^2`."""

        # This integral is easier to compute with respect to original parametrization.
        # Wolfram Alpha gives us the answer
        q_1, h, D = self.original_parametrization()
        term = np.exp(((h - 2) / h) * np.log1p(D * h * t))
        return term * q_1**2 / (D * (h - 2))

    def eval_integral_from_0_to_inf(self):
        r"""Evaluate
        :math:`\int_0^{\infty} \mu(t; \boldsymbol{\theta}) dt`."""
        t1 = self.theta1
        return np.exp(t1)

    def eval_integral_from_0_to_T(self, T):
        r"""Evaluate
        :math:`\int_0^{T} \mu(t; \boldsymbol{\theta}) dt`."""
        t1 = self.theta1
        t2 = self.theta2
        t3 = self.theta3
        logterm = log1pexp(t=T, x=t3 - t2)
        return np.exp(t1) - np.exp(t1 - np.exp(-t3) * logterm)

    def eval_integral_from_T_to_inf(self, T):
        r"""Evaluate
        :math:`\int_T^{\infty} \mu(t; \boldsymbol{\theta}) dt`."""
        t1 = self.theta1
        t2 = self.theta2
        t3 = self.theta3
        logterm = log1pexp(t=T, x=t3 - t2)
        return np.exp(t1 - np.exp(-t3) * logterm)

    def eval_integral_from_T1_to_T2(self, T1, T2):
        r"""Evaluate
        :math:`\int_{T_1}^{T_2} \mu(t; \boldsymbol{\theta}) dt`."""
        high = self.eval_integral_from_0_to_T(T2)
        low = self.eval_integral_from_0_to_T(T1)
        return high - low

    def eval_grad_log(self, t):
        r"""Evaluate the gradient of the log-Arps curve.

        Returns
        -------
        np.ndarray
            Array of shape (3, len(t)) with gradient information.
            To compute the gradient with respect to the parameters
            :math:`\boldsymbol{\theta}`, the caller must sum over :math:`t`.

        Notes
        -----

        This method evaluates the gradient of the log-Arps curve:


        .. math::

            \nabla_{\boldsymbol{\theta}} \log \mu(t; \boldsymbol{\theta}) =
            \frac{1}{\mu(t; \boldsymbol{\theta})}
            \nabla_{\boldsymbol{\theta}} \mu(t; \boldsymbol{\theta})

        Note that this method can be used to evaluate the gradient of the Arps
        curve, since:

        .. math::

            \nabla_{\boldsymbol{\theta}} \mu(t; \boldsymbol{\theta}) =
            \mu(t; \boldsymbol{\theta})
            \nabla_{\boldsymbol{\theta}} \log \mu(t; \boldsymbol{\theta})

        Examples
        --------
        >>> model = Arps(1, -2, 1)
        >>> time = np.arange(5)
        >>> gradient = model.eval_grad_log(time)
        >>> gradient.shape
        (3, 5)
        >>> gradient.sum(axis=1)
        array([5.        , 0.334..., 0.286...])
        """
        t2 = self.theta2
        t3 = self.theta3

        # Differentiate the expression for log(arps(t, theta)) =
        # t1 - t2 - (1 + exp(-t3)) * log1p(t * exp(t3 - t2)).
        # These computations can be checked with e.g. Wolfram Alpha
        grad_t1 = np.ones_like(t)
        grad_t2 = (t - np.exp(t2)) / (t * np.exp(t3) + np.exp(t2))
        grad_t3 = np.exp(-t3) * log1pexp(t=t, x=t3 - t2)
        grad_t3 -= (np.exp(t3 - t2) * (np.exp(-t3) + 1) * t) / (t * np.exp(t3 - t2) + 1)

        return np.array([grad_t1, grad_t2, grad_t3])

    def original_parametrization(self):
        r"""Return parameters :math:`(q_1, h, D)` in the original
        parametrization of the Arps curve, given by:

        .. math::

            \mu(t; q_1, h, D) = \frac{q_1}{(1 + hDt)^{1/h}}

        Examples
        --------
        >>> Arps(1, -2, 1).original_parametrization()
        (20.08..., 0.73..., 27.47...)

        """
        # M = q1/((1 - h)D), b = 1/((1 - h)D), and k = h/(1-h), we
        h = logistic_sigmoid(self.theta3)
        D = 1 / ((1 - h) * np.exp(self.theta2))
        q_1 = np.exp(self.theta1) * (1 - h) * D
        return float(q_1), float(h), float(D)

    @staticmethod
    def eval_log_original_parametrization(t, q_1, h, D):
        r"""Evaluate :math:`\log \mu(t; q_1, h, D)`."""
        # log (arps(t; q1, h, D)) := log(q_1) - log1p(h*D*t)/h
        return np.log(q_1) - np.log1p(h * D * t) / h

    @staticmethod
    def eval_original_parametrization(t, q_1, h, D):
        r"""Evaluate :math:`\mu(t; q_1, h, D)`."""
        return np.exp(np.log(q_1) - np.log1p(h * D * t) / h)

    @classmethod
    def from_original_parametrization(cls, q_1, h, D):
        """Initialize an instance using the original parametrization."""
        assert 0 < h < 1
        assert q_1 > 0
        assert D > 0
        # Convert from Arps parametrization to Generalized Pareto Distribution
        # parametrization. See bottom of page 11 in the paper by Lee et al
        b = 1 / ((1 - h) * D)
        M = b * q_1
        # k = h / (1 - h)
        # Page 15 in the paper by Lee et al
        theta1 = np.log(M)
        theta2 = np.log(b)
        theta3 = np.log(h) - np.log1p(-h)
        return cls(theta1=theta1, theta2=theta2, theta3=theta3)


@dataclasses.dataclass
class Constant:
    r"""Create a Constant instance.

    Parameters
    ----------
    theta1 : float
        The one and only parameter in the model.

    Notes
    -----

    The Constant class holds a constant model, which is only used for testing
    purposes and sanity checks.

    .. math::

        \log \left( \mu(t; \theta_1) \right) = \theta_1

    Examples
    --------
    >>> model = Constant(1)
    >>> time = np.arange(5)
    >>> model(time)
    array([2.71828183, 2.71828183, 2.71828183, 2.71828183, 2.71828183])
    """

    def __init__(self, theta1: float):
        self.theta1 = float(theta1)

    def original_parametrization(self):
        r"""Return parameter :math:`C` in the original
        parametrization of the Constant curve, given by:

        .. math::

            \mu(t; C) = C

        Examples
        --------
        >>> time = np.array([0, 1, 2, 3, 4])
        >>> C = 10
        >>> curve = Constant.from_original_parametrization(C)
        >>> curve.eval(time)
        array([10., 10., 10., 10., 10.])
        """
        return (float(np.exp(self.theta1)),)

    @classmethod
    def from_original_parametrization(cls, C):
        """Initialize an instance using the original parametrization.

        Examples
        --------
        >>> C = 1
        >>> curve = Constant.from_original_parametrization(C)
        >>> curve
        Constant(0.0,)
        >>> curve.original_parametrization()
        (1.0,)
        """
        assert C > 0
        return cls(theta1=np.log(C))

    def __repr__(self):
        return f"Constant{self.thetas}"

    @property
    def thetas(self):
        """Return the thetas as a tuple."""
        return (self.theta1,)

    def eval_log(self, t):
        r"""Evaluate :math:`\log \mu(t; \boldsymbol{\theta})`."""
        t = np.array(t)
        assert np.all(np.isfinite(t)), "All values must be finite and not NaN"
        return np.ones_like(t) * self.theta1

    def eval(self, t):
        r"""Evaluate :math:`\mu(t; \boldsymbol{\theta})`."""
        return np.exp(self.eval_log(t))

    def __call__(self, t):
        return self.eval(t)

    def add(self, k):
        """Add a constant k in log space and return a new instance.

        Examples
        --------
        >>> model = Constant(10)
        >>> model.add(2)
        Constant(12.0,)
        """
        t1 = self.theta1 + k
        return type(self)(t1)

    def shift_and_scale(self, shift=0, scale=1):
        r"""Scale and shift the curve along the horizontal axis.

        Equivalent to fitting on a transformed grid:
            :math:`t' = t \times \text{scale} + \text{shift}`.

        Examples
        --------
        >>> curve = Constant(0)
        >>> x_grid = np.array([0, 1, 2, 3, 5])
        >>> curve.eval(x_grid)
        array([1., 1., 1., 1., 1.])
        >>> curve.shift_and_scale(scale=2, shift=3).eval((x_grid * 2 + 3))
        array([1., 1., 1., 1., 1.])
        """
        assert isinstance(shift, numbers.Real)
        assert isinstance(scale, numbers.Real)
        assert scale > 0, "Scale must be positive"
        return self

    def eval_integral(self, t):
        r"""Evaluate the anti-derivative of :math:`\mu(t; \boldsymbol{\theta})`."""
        return np.exp(self.theta1) * t

    def eval_squared_integral(self, t):
        r"""Evaluate the anti-derivative of :math:`\mu(t; \boldsymbol{\theta})^2`."""
        return np.exp(2 * self.theta1) * t

    def eval_integral_from_0_to_inf(self):
        r"""Evaluate
        :math:`\int_0^{\infty} \mu(t; \boldsymbol{\theta}) dt`."""
        return np.inf

    def eval_integral_from_0_to_T(self, T):
        r"""Evaluate
        :math:`\int_0^{T} \mu(t; \boldsymbol{\theta}) dt`."""
        return np.exp(self.theta1) * T

    def eval_integral_from_T_to_inf(self, T):
        r"""Evaluate
        :math:`\int_T^{\infty} \mu(t; \boldsymbol{\theta}) dt`."""
        return np.inf

    def eval_integral_from_T1_to_T2(self, T1, T2):
        r"""Evaluate
        :math:`\int_{T_1}^{T_2} \mu(t; \boldsymbol{\theta}) dt`."""
        high = self.eval_integral_from_0_to_T(T2)
        low = self.eval_integral_from_0_to_T(T1)
        return high - low

    def eval_grad_log(self, t):
        r"""Evaluate the gradient of the log-Constant curve.

        Returns
        -------
        np.array
            Array of shape (1, len(t)) with gradient information.
            To compute the gradient with respect to the parameters
            :math:`\boldsymbol{\theta}`, the caller must sum over :math:`t`.

        Examples
        --------
        >>> model = Constant(5)
        >>> time = np.arange(5)
        >>> gradient = model.eval_grad_log(time)
        >>> gradient.shape
        (1, 5)
        >>> gradient
        array([[1, 1, 1, 1, 1]])
        """
        return np.ones_like(t).reshape(1, -1)


@dataclasses.dataclass
class Exponential:
    r"""Create an Exponential instance.

    Parameters
    ----------
    theta1 : float
        Parameter number one. Interpreted as the logarithm of the integral from
        zero to infinity. This scales the exponential distribution.
    theta2 : float
        Parameter number two. This is the rate parameter. Compared to Wikipedia's
        article on the exponential distribution, lambda = exp(theta2).

    Notes
    -----

    The Exponential class holds an exponential curve, :math:`\mu(t; \boldsymbol{\theta})`
    given by,

    .. math::

        \log \left( \mu(t; \theta_1, \theta_2) \right) = \theta_1 - \theta_2 - t \exp(-\theta_2),

    which assures that the curve is positive and decreasing.


    See also:
        https://en.wikipedia.org/wiki/Exponential_distribution


    Examples
    --------
    >>> model = Exponential(1, -1)
    >>> time = np.arange(5)
    >>> model(time)
    array([7.38905610e+00, 4.87589299e-01, 3.21750601e-02, 2.12316902e-03,
           1.40103753e-04])

    Verify that Exponential(theta1, theta2) == Arps(theta1, theta2, -∞):

    >>> import numpy as np
    >>> exponential = Exponential(1, -1.3)
    >>> arps = Arps(1, -1.3, -20)  # Small value of theta3
    >>> np.allclose(exponential(time), arps(time))
    True

    This example shows the difference between midpoint evaluation and integrating
    the exponential decline curve within each time period. Since the decline
    curve is convex, using the midpoint rule underestimates, but very slightly:

    >>> exponential = Exponential.from_original_parametrization(C=1, k=0.1)
    >>> time_on = np.array([1, 0.5, 1, 0.4, 1])
    >>> midpoints = np.cumsum(time_on) - time_on / 2
    >>> exponential.eval(midpoints) * time_on
    array([0.95122942, 0.44124845, 0.81873075, 0.3053518 , 0.71177032])
    >>> T2 = np.cumsum(time_on)
    >>> T1 = T2 - time_on
    >>> exponential.eval_integral_from_T1_to_T2(T1, T2)
    array([0.95162582, 0.44129442, 0.81907193, 0.30537215, 0.71206693])
    """

    def __init__(self, theta1: float, theta2: float):
        self.theta1 = float(theta1)
        self.theta2 = float(theta2)

    def original_parametrization(self):
        r"""Return parameters :math:`(C, k)` in the original
        parametrization of the Exponential curve, given by:

        .. math::

            \mu(t; C, k) = C \exp(-k t)

        Examples
        --------
        >>> time = np.linspace(0, 10)
        >>> C, k = 10, 0.1
        >>> curve = Exponential.from_original_parametrization(C, k)
        >>> np.allclose(C * np.exp(-k * time), curve(time))
        True
        """
        t1 = self.theta1
        t2 = self.theta2
        return float(np.exp(t1 - t2)), float(np.exp(-t2))

    @classmethod
    def from_original_parametrization(cls, C, k):
        """Initialize an instance using the original parametrization.

        Examples
        --------
        >>> C, k = 10, 0.1
        >>> theta1, theta2 = Exponential.from_original_parametrization(C, k).thetas
        >>> Exponential(theta1, theta2).original_parametrization()
        (10.0000000..., 0.1000000...)
        """
        assert C > 0
        assert k > 0
        theta2 = -np.log(k)
        theta1 = np.log(C / k)
        return cls(theta1=theta1, theta2=theta2)

    def __repr__(self):
        return f"Exponential{self.thetas}"

    @property
    def thetas(self):
        """Return the thetas as a tuple."""
        return (self.theta1, self.theta2)

    def eval_log(self, t):
        r"""Evaluate :math:`\log \mu(t; \boldsymbol{\theta})`."""
        t = np.array(t)
        assert np.all(np.isfinite(t)), "All values must be finite and not NaN"
        t1 = self.theta1
        t2 = self.theta2
        return t1 - t2 - t * np.exp(-t2)

    def eval(self, t):
        r"""Evaluate :math:`\mu(t; \boldsymbol{\theta})`."""
        return np.exp(self.eval_log(t))

    def __call__(self, t):
        return self.eval(t)

    def add(self, k):
        """Add a constant k in log space and return a new instance.

        Examples
        --------
        >>> model = Exponential(10, 3)
        >>> model.add(2)
        Exponential(12.0, 3.0)
        """
        t1 = self.theta1 + k
        t2 = self.theta2
        return type(self)(t1, t2)

    def shift_and_scale(self, shift=0, scale=1):
        r"""Scale and shift the curve along the horizontal axis.

        Equivalent to fitting on a transformed grid:
            :math:`t' = t \times \text{scale} + \text{shift}`.

        Examples
        --------
        >>> curve = Exponential(10, 1)
        >>> x_grid = np.array([0, 1, 2, 3, 5])
        >>> curve.eval_log(x_grid)
        array([9.        , 8.63212056, 8.26424112, 7.89636168, 7.16060279])
        >>> curve.shift_and_scale(scale=2, shift=3).eval_log((x_grid * 2 + 3))
        array([9.        , 8.63212056, 8.26424112, 7.89636168, 7.16060279])
        """
        assert isinstance(shift, numbers.Real)
        assert isinstance(scale, numbers.Real)
        assert scale > 0, "Scale must be positive"

        scale, shift = 1 / scale, -shift / scale

        # Algebra :)
        theta1 = self.theta1 - shift * np.exp(-self.theta2) - np.log(scale)
        theta2 = self.theta2 - np.log(scale)

        return type(self)(theta1=theta1, theta2=theta2)

    def eval_integral(self, t):
        r"""Evaluate the anti-derivative of :math:`\mu(t; \boldsymbol{\theta})`."""
        t1 = self.theta1
        t2 = self.theta2
        return -np.exp(t1 - t * np.exp(-t2))

    def eval_squared_integral(self, t):
        r"""Evaluate the anti-derivative of :math:`\mu(t; \boldsymbol{\theta})^2`."""
        t1 = self.theta1
        t2 = self.theta2
        exponent = 2 * t1 - 2 * t * np.exp(-t2) - t2
        return -np.exp(exponent) / 2

    def eval_integral_from_0_to_inf(self):
        r"""Evaluate
        :math:`\int_0^{\infty} \mu(t; \boldsymbol{\theta}) dt`."""
        t1 = self.theta1
        return np.exp(t1)

    def eval_integral_from_0_to_T(self, T):
        r"""Evaluate
        :math:`\int_0^{T} \mu(t; \boldsymbol{\theta}) dt`."""
        t1 = self.theta1
        t2 = self.theta2
        return np.exp(t1) * (1 - np.exp(-T * np.exp(-t2)))

    def eval_integral_from_T_to_inf(self, T):
        r"""Evaluate
        :math:`\int_T^{\infty} \mu(t; \boldsymbol{\theta}) dt`."""
        t1 = self.theta1
        t2 = self.theta2
        exponent = t1 - np.exp(-t2) * T
        return np.exp(exponent)

    def eval_integral_from_T1_to_T2(self, T1, T2):
        r"""Evaluate
        :math:`\int_{T_1}^{T_2} \mu(t; \boldsymbol{\theta}) dt`."""
        high = self.eval_integral_from_0_to_T(T2)
        low = self.eval_integral_from_0_to_T(T1)
        return high - low

    def eval_grad_log(self, t):
        r"""Evaluate the gradient of the log-Exponential curve.

        Returns
        -------
        np.array
            Array of shape (2, len(t)) with gradient information.
            To compute the gradient with respect to the parameters
            :math:`\boldsymbol{\theta}`, the caller must sum over :math:`t`.

        Notes
        -----

        This method evaluates the gradient of the log-Exponential curve:

        .. math::

            \nabla_{\boldsymbol{\theta}} \log \mu(t; \boldsymbol{\theta}) =
            \frac{1}{\mu(t; \boldsymbol{\theta})}
            \nabla_{\boldsymbol{\theta}} \mu(t; \boldsymbol{\theta})

        Note that this method can be used to evaluate the gradient of the
        Exponential curve, since:

        .. math::

            \nabla_{\boldsymbol{\theta}} \mu(t; \boldsymbol{\theta}) =
            \mu(t; \boldsymbol{\theta})
            \nabla_{\boldsymbol{\theta}} \log \mu(t; \boldsymbol{\theta})

        Examples
        --------
        >>> model = Exponential(5, 0.66)
        >>> time = np.arange(5)
        >>> gradient = model.eval_grad_log(time)
        >>> gradient.shape
        (2, 5)
        >>> gradient.sum(axis=1)
        array([5.        , 0.16851334])
        """

        t2 = self.theta2

        # Differentiate the expression for log(exponential(t, theta)) =
        # t1 - t2 - t * exp(-t2)
        # These computations can be checked with e.g. Wolfram Alpha
        grad_t1 = np.ones_like(t)
        grad_t2 = t * np.exp(-t2) - 1

        return np.array([grad_t1, grad_t2])


def softplus(x):
    r"""Numerically stable implementation of :math:`\log (1 + \exp(x))`.

    domain(x) = [-inf, inf]
    Source:
    https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python

    Examples
    --------
    >>> float(softplus(0))
    0.69314...
    >>> float(softplus(100)) # This would overflow with a naive implementation
    100.0
    >>> float(softplus(-100))
    3.720...e-44
    """
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_inv(x):
    """Numerically stable implementation of log(exp(x) - 1).

    domain(x) = (0, inf]

    Examples
    --------
    >>> float(softplus(softplus_inv(1000)))
    1000.0
    >>> float(softplus_inv(softplus(99)))
    99.0
    """
    return np.log1p(-np.exp(-np.abs(x))) + np.maximum(x, 0)


def log1pexp(t, x):
    """Implements log(1 + t * np.exp(x)) in a numerically stable way.

    See:
    https://cs.stackexchange.com/questions/110798/numerically-stable-log1pexp-calculation

    Parameters
    ----------
    t : np.ndarray or number
    x : float

    Returns
    -------
    np.ndarray or number
        The expression log(1 + t * np.exp(x)).

    Examples
    --------
    >>> t = 1
    >>> float(log1pexp(t, x=10))
    10.000045398899...
    >>> float(log1pexp(t, x=1000))
    1000.0
    >>> float(log1pexp(t=200, x=1000))
    1005.2983173665...
    >>> t, x = -1, -1
    >>> bool(np.isclose(log1pexp(t, x), np.log(1 + t * np.exp(x))))
    True

    """
    assert not isinstance(x, np.ndarray), "x must be a number"

    # The trick, as explained in:
    # https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    # is to write: log(1 + t * exp(x)) = log(1 + t * exp(x)) - log(exp(x)) + x
    #  = log((1 + t * exp(x)) / exp(x)) + x = log((1 + t * exp(x)) / exp(x)) + x
    #  = log(exp(-x) + t) + x
    # When x is large, we should use this second representation

    # If t is an array
    if isinstance(t, np.ndarray):
        # The expression exp(x) overflows if x > 10**2.85, but we
        # set the threshold at x=0 for a nice symmetry
        if x > 0:
            return np.log(np.exp(-x) + t) + x
        else:
            return np.log1p(t * np.exp(x))

    # If t is a number, convert to array, call the function and return
    return log1pexp(np.array([t]), x)[0]


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
