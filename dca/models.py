"""
Models
------

Models that use a decline curve analysis function (e.g. Arps) to simulate data,
define a likelihood function that can be optimized, and the optimization
routines.

A brief note on the history of our models used for DCA:

    - For a long time we thought of curve fitting as a ML problem, minimizing
      a loss function without giving it a statistical interpretation.
      The loss function was:
          sum_i w(t_i) |log(y_i) - log(f(t_i; theta))|^p + prior(theta)
    - Then we added a statistical interpretation to the loss function above,
      by minimizing the negative log-likelihood of a generalized normal (GND):
          sum_i w(t_i) GND(mu=log(f(t_i; theta)),
                           alpha=sqrt(tau_i) * sigma,
                           beta=p).logpdf(log(y_i))
                + prior(theta)
      This is essentially the same model, but with a sigma term for independent errors.
    - Finally, when we started running simulations we saw that the assumption
      of independent errors does not hold. Errors are dependent. Typically a well
      can deviate from the trend f(t_i; theta) for many months. If we ignore
      autocorrelation we grossly underestimate the uncertainty, because in our
      simulations the independent errors cancel each other out quickly.
      To alleviate this we added a phi parameter for autocorrelation.

Note that setting phi=0 on the final model recovers the model with independent
errors. Setting phi=0 and sigma=1 recovers the original loss (to to a constant).
Finally, setting beta=p=2.0 and w(t_i) = 1 recovers least squares regression.
"""

import functools
import inspect
import itertools
import numbers

import numpy as np
import scipy as sp

from dca.decline_curve_analysis import Arps, Constant, Exponential
from dca.optimization import Optimizer, Parameter
from dca.utils import weighted_linregress


class Prior:
    """A default prior, used if no prior is given. A user specified prior must
    also comply with this API.
    """

    def __init__(self, theta_mean, phi_v_b=100):
        # Set up prior distributions once here, since this is surprisingly slow.
        # In fact, it was a bottleneck: setting up a single sp.stats.distr takes
        # ~700 μs, but evaluating the rest of the prior (__call__) takes ~329 μs
        self.theta_mean = np.array(theta_mean)
        self.theta_prior = sp.stats.multivariate_normal(mean=theta_mean)
        self.phi_v_b = phi_v_b

    def __call__(self, theta, sigma, phi, p, prior_strength):
        """A default prior, used if no prior is given. Returns (value, gradient)
        and must, for any parameters in the domain, return a finite answer.

        Examples
        --------
        >>> default_prior = Prior(theta_mean=np.zeros(2))
        >>> value = default_prior(theta=(1, 2), sigma=1, phi=1.0,
        ...                       p=1.0, prior_strength=1.0)
        >>> float(value)
        2469.05...
        """
        theta = np.array(theta)

        # At the boundary the prior evaluates to infty, so we clip the values
        epsilon = 1e-10
        sigma = sigma if sigma is None else max(0 + epsilon, sigma)
        phi = phi if phi is None else min(max(0 + epsilon, phi), 1 - epsilon)
        p = p if p is None else min(max(1 + epsilon, p), 2 - epsilon)

        # Values here must match the gradient
        theta_v = self.theta_prior.logpdf(theta)
        sigma_v = gamma_logpdf(sigma, a=1.5, scale=1)
        # If we do not regularize heavily as phi->1, then the model is not
        # identifiable. Low negative log-likelihood can be achieved with either:
        #   (1) a good decline curve and reasonable autocorrelation or
        #   (2) ANY decline curve and a huge autocorrelation.
        # This avoid the latter case, where the whole trend is interpreted as
        # very autocorrelated noise, we must regularize.
        phi_v = beta_logpdf(phi, a=2, b=self.phi_v_b)
        p_v = beta_logpdf(p - 1, a=10, b=10)

        # Compute the negative log pdf
        neg_log_pdf = -(prior_strength * theta_v + sigma_v + phi_v + p_v)
        return neg_log_pdf

    def gradient(self, theta, sigma, phi, p, prior_strength):
        """A default prior, used if no prior is given. Returns (value, gradient)
        and must, for any parameters in the domain, return a finite answer.

        Examples
        --------
        >>> default_prior = Prior(theta_mean=np.zeros(2))
        >>> gradient = default_prior.gradient(theta=(1, 2), sigma=1, phi=1.0,
        ...                                   p=1.0, prior_strength=1.0)
        >>> gradient
        array([ 1.00000000e+00,  2.00000000e+00,  5.00000000e-01,  9.89999918e+11,
               -8.99999925e+10])
        """
        theta = np.array(theta)

        # At the boundary the prior evaluates to infty, so we clip the values
        epsilon = 1e-10
        sigma = sigma if sigma is None else max(0 + epsilon, sigma)
        phi = phi if phi is None else min(max(0 + epsilon, phi), 1 - epsilon)
        p = p if p is None else min(max(1 + epsilon, p), 2 - epsilon)

        # Gradients of parameters (values here must match those in logpdf)
        grad_theta = self.theta_mean - theta
        grad_sigma = gamma_logpdf_grad(sigma, a=1.5, scale=1)
        grad_phi = beta_logpdf_grad(phi, a=2, b=self.phi_v_b)
        grad_p = beta_logpdf_grad(p - 1, a=10, b=10)

        # All gradients: numbers, arrays
        gradients = [prior_strength * grad_theta, grad_sigma, grad_phi, grad_p]
        # Flatten: [2, [3, 4]] => [2, 3, 4]
        chained = list(itertools.chain(*(np.atleast_1d(g) for g in gradients)))
        return -np.array(chained)


def beta_logpdf_grad(x, *, a, b):
    """Compute the derivative of the log-pdf of beta with respect to x.

    Examples
    --------
    >>> from scipy.stats import beta
    >>> beta_distr = beta(a=2, b=4.33)
    >>> eps = 1e-4
    >>> numerical_derivative = (beta_distr.logpdf(0.5 + eps) -
    ...                         beta_distr.logpdf(0.5 - eps)) / (2 * eps)
    >>> float(numerical_derivative)
    -4.66...
    >>> beta_logpdf_grad(x=0.5, a=2, b=4.33)
    -4.66...
    """
    return (a - 1) / x - (b - 1) / (1 - x)


def beta_logpdf(x, *, a, b):
    """Compute the log-pdf of beta with respect to x, faster than scipy.

    Examples
    --------
    >>> from scipy.stats import beta
    >>> beta_distr = beta(a=2, b=4.33)
    >>> float(beta_distr.logpdf(0.33))
    0.69666585...
    >>> float(beta_logpdf(x=0.33, a=2, b=4.33))
    0.69666585...
    """
    logpdf = sp.special.xlog1py(b - 1.0, -x) + sp.special.xlogy(a - 1.0, x)
    return logpdf - sp.special.betaln(a, b)


def gamma_logpdf(x, *, loc=0, scale=1, a=1):
    """Compute the log-pdf of gamma with respect to x, faster than scipy.

    Examples
    --------
    >>> from scipy.stats import gamma
    >>> gamma_distr = gamma(loc=1, scale=0.33, a=0.5)
    >>> float(gamma_distr.logpdf(2))
    -3.04833666...
    >>> float(gamma_logpdf(2, loc=1, scale=0.33, a=0.5))
    -3.04833666...
    """
    y = (x - loc) / scale
    return sp.special.xlogy(a - 1.0, y) - y - sp.special.gammaln(a) - np.log(scale)


def gamma_logpdf_grad(x, *, loc=0, scale=1, a=1):
    """Compute the derivative of the log-pdf of gamma with respect to x.

    Examples
    --------
    >>> from scipy.stats import gamma
    >>> gamma_distr = gamma(loc=1, scale=0.33, a=0.5)
    >>> eps = 1e-6
    >>> numerical_derivative = (gamma_distr.logpdf(2 + eps) -
    ...                         gamma_distr.logpdf(2 - eps)) / (2 * eps)
    >>> float(numerical_derivative)
    -3.5303030...
    >>> gamma_logpdf_grad(x=2, loc=1, scale=0.33, a=0.5)
    -3.5303030...
    """
    return (a - 1) / (x - loc) - 1 / scale


def gennorm_std(*, scale, beta):
    """Computes the std of the generalized normal.

    This approach is faster than sp.stats.gennorm.std(), which is a bottleneck
    in the log-likelihood .

    Examples
    --------
    >>> std1 = sp.stats.gennorm(scale=3.15, beta=1.5).std()
    >>> std2 = gennorm_std(scale=3.15, beta=1.5)
    >>> bool(np.isclose(std1, std2))
    True
    """
    return scale * np.sqrt(sp.special.gamma(3 / beta) / sp.special.gamma(1 / beta))


def gennorm_logpdf(x, *, loc, scale, beta):
    """Equivalent to calling sp.stats.gennorm(loc, scale, beta).logpdf(x),
    but around 10 times faster.

    Examples
    --------
    >>> x = np.array([0, 0.5, 1, 1.5])
    >>> sp.stats.gennorm(loc=1, scale=2, beta=1.5).logpdf(x)
    array([-1.63753292, -1.40897953, -1.28397953, -1.40897953])
    >>> gennorm_logpdf(x, loc=1, scale=2, beta=1.5)
    array([-1.63753292, -1.40897953, -1.28397953, -1.40897953])
    """
    # https://en.wikipedia.org/wiki/Generalized_normal_distribution
    y = (x - loc) / scale
    # It's 1/3 faster to compute np.abs(y) ** beta this way, and ** is slow
    abs_y_pow_beta = np.exp(beta * np.log(np.abs(y) + 1e-36))
    return np.log(0.5 * beta / scale) - sp.special.gammaln(1.0 / beta) - abs_y_pow_beta


class AR1Model:
    r"""Class for an autoregressive regression model, consisting of:

    1. A non linear-regression and
    2. errors that follow an AR(1) model

    We need two or three things to make a statistical model like this work:

    - Simulation
    - Likelihood
    - Gradients (not needed, but makes optimizer hyperparams faster/robust)

    In probabilistic programming languages like Stan, you only need to specify
    the forward simulation model. Stan will infer the likelihood and the
    gradients automatically. Stan also has the option of running MCMC instead
    of optimizing. Using Stan instead of writing out models "by hand" could be
    a good option, and samples from the posterior via MCMC could be useful.
    We chose not to use Stan because of difficulties with C++ compiles across
    platforms, setting up environments on Windows, etc. Maybe in the future!

    An important thing to know about this model is that it underspecified
    without a prior on the autocorrelation coefficient phi. A declining trend
    can either be interpreted as:

    1. an actual declining trend, with some autocorrlated noise on top
    2. an arbitrary trend, with huge autocorrelation

    In case (2), the whole trend is interpreted as one big autocorrelated noise
    event. This happens as phi goes to one (typically phi>0.99), and it's not
    what we want. It's important to regularize phi with an appropriate prior.

    References
    ----------
    - https://en.wikipedia.org/wiki/Autoregressive_model
    - https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation
    - https://en.wikipedia.org/wiki/Prais%E2%80%93Winsten_estimation

    Examples
    --------
    >>> from dca import Arps
    >>> model = AR1Model(curve_cls=Arps)
    >>> model = model.update(theta=(5, 3, 1), sigma=1, phi=0.5, p=2)
    >>> tau = np.ones(10)
    >>> t = np.linspace(0, 10, num=10)
    >>> simulations = model.simulate(tau=tau, t=t, seed=42, simulations=5)
    >>> simulations.shape
    (5, 10)

    A good and a bad guess:

    >>> float(model.negative_ll(theta=(5, 3, 1), sigma=1, phi=0.5, p=2, tau=tau,
    ...                   t=t, y=simulations.mean(axis=0)).sum())
    7.35...
    >>> float(model.negative_ll(theta=(0, 3, 1), sigma=1, phi=0.5, p=2, tau=tau,
    ...                   t=t, y=simulations.mean(axis=0)).sum())
    89.6...
    """

    def __init__(self, curve_cls):
        assert curve_cls in (Arps, Exponential, Constant)

        self.curve_cls = curve_cls
        self.n_thetas = len(inspect.Signature.from_callable(self.curve_cls).parameters)

        # If these are set to None, they are free parameters to be optimized
        self.theta = None
        self.phi = None
        self.sigma = None
        self.p = None

    def update(self, **kwargs):
        """Update parameters (theta, sigma, phi, p) to values on the original,
        bounded scale. Setting them to None means they will be inferred by optimization."""
        for var_name, var_value in kwargs.items():
            setattr(self, var_name, var_value)
        return self

    def pointwise_std(self, *, p, sigma, phi):
        r"""Compute the pointwise standard deviation.

        .. math::

            \frac{\sqrt{\Gamma(3 / \beta)}}{\sqrt{\Gamma(1 / \beta)}} \frac{\sigma}{\sqrt{1 - \phi^2}}

        See:

        - https://en.wikipedia.org/wiki/Autoregressive_model#Example:_An_AR(1)_process
        - https://en.wikipedia.org/wiki/Generalized_normal_distribution
        """
        return float(gennorm_std(beta=p, scale=sigma) / np.sqrt(1 - phi**2))

    def simulate(self, *, tau, t, seed=None, prev_eta=None, simulations=1):
        r"""Simulate production rates :math:`y` from the model:
        
        .. math::
        
            \log(y_i) &= \log(f(t_i)) + \eta_i \\
            \quad \eta_i &= \phi \cdot \eta_{i-1} + \sqrt{\tau_i} \cdot \epsilon_i \\
            \quad \epsilon_i &\sim \text{GN}(\mu=0, \alpha=\sigma, \beta)
        
        where:
        
        - :math:`y_i = \frac{p_i}{\tau_i} = \text{production per time}`
        - :math:`\tau_i = \text{time_on}`
        - GN is the Generalized Normal with p-norm given by :math:`\beta`
        
        To get simulated production, multiply by :math:`\text{time_on}`.
        
        The returned numpy array has shape (simulations, len(x)).
        """
        assert self.theta is not None, "All parameters must be fixed to simulate"
        assert self.phi is not None, "All parameters must be fixed to simulate"
        assert self.sigma is not None, "All parameters must be fixed to simulate"
        assert self.p is not None, "All parameters must be fixed to simulate"
        assert len(tau) == len(t)
        assert np.all(tau > 0)
        assert np.all(np.diff(t) > 0)
        rng = np.random.default_rng(seed)
        curve = self.curve_cls(*self.theta)

        gennorm = sp.stats.gennorm(beta=self.p, scale=self.sigma)

        if prev_eta is None:
            # We assume that the long-running process is normally distributed
            # with variance equal to var_epsilon / (1 - phi**2), where
            # var_epsilon is the variance of the generalized normal noise.

            # Compute the long-running std
            scale = gennorm.std() / np.sqrt(1 - self.phi**2)
            prev_eta = rng.normal(scale=scale, size=simulations)

        shape = (simulations, len(t))
        log_y = np.zeros(shape=shape)
        epsilon_ij = gennorm.rvs(size=shape, random_state=rng)

        for i in range(len(t)):
            # Draw noise epsilon
            epsilon_i = epsilon_ij[:, i]

            # Correlated errors eta
            eta_i = self.phi * prev_eta + np.sqrt(tau[i]) * epsilon_i
            log_y[:, i] = curve.eval_log(t[i]) + eta_i
            prev_eta = eta_i

        return np.exp(log_y)

    def _initial_guess(self, *, t, y, tau, half_life):
        """Solves a simpler problem with:

        - no autocorrelation (phi=0)
        - normal errors (p=beta=2)

        This helps get a decent initial guess for parameters."""
        assert len(t) == len(y) == len(tau)

        # Regress a linear function on log(y)
        if self.curve_cls in (Exponential, Arps):
            # Exponential(C, k) is linear in log space, so we do a linreg
            w = self.get_weights(t=t, tau=tau, half_life=half_life)
            slope, intercept = weighted_linregress(x=t, y=np.log(y), w=w)
            # Enforce negative slope and positive intercept
            slope, intercept = min(-1e-6, slope), max(1e-6, intercept)
            C, k = np.exp(intercept), -slope  # Transform parameters
            curve = Exponential.from_original_parametrization(C, k)

            residuals = np.log(y) - curve.eval_log(t)
            w_normalized = w / w[-1]
            std = np.sqrt(np.sum(w_normalized * residuals**2) / np.sum(w_normalized))

            theta = np.zeros(self.n_thetas) - 2
            theta[:2] = curve.thetas

        elif self.curve_cls == Constant:
            w = self.get_weights(t=t, tau=tau, half_life=half_life)
            theta1 = np.average(np.log(y), weights=w)

            residuals = np.log(y) - theta1
            w_normalized = w / w[-1]
            std = np.sqrt(np.sum(w_normalized * residuals**2) / np.sum(w_normalized))

            theta = np.array([theta1])

        # Here we assume p and phi, then adjust the uncorrelated std estimated
        # above to account for autocorrelation
        p = 1.5
        phi = 0.66
        sigma = max(std * np.sqrt(1 - phi**2), 0.01)  # Cannot be too small

        return {
            "theta": tuple(float(t) for t in theta),
            "sigma": float(sigma),
            "phi": phi,
            "p": p,
        }

    def optimize(
        self, *, tau, t, y, prior_strength=0.0, half_life=np.inf, neg_logpdf_prior=None
    ):
        """Optimize over parameters (theta, sigma, phi, p)."""
        params = ["theta", "phi", "sigma", "p"]
        assert any(getattr(self, param) is None for param in params), (
            "At least one param must be free to optimize"
        )
        assert len(tau) == len(t) == len(y)
        assert np.all(y > 0)
        assert np.all(tau > 0)
        assert np.all(np.diff(t) > 0)
        assert prior_strength >= 0
        assert half_life >= 0
        assert (neg_logpdf_prior is None) or callable(neg_logpdf_prior)

        # The prior function
        if neg_logpdf_prior is None:
            neg_logpdf_prior = Prior(theta_mean=np.zeros(self.n_thetas))

        # Convert to arrays and make a copy
        tau, t, y = np.array(tau), np.array(t), np.array(y)

        def posterior(
            theta, sigma, phi, p, *, tau, t, y, half_life=np.inf, prev_eta=None
        ) -> float:
            """Evaluate the negative log posterior distribution.
            log (posterior) = log(prior) + log(likelihood)"""
            likelihood = self.negative_ll(
                theta=theta,
                sigma=sigma,
                phi=phi,
                p=p,
                tau=tau,
                t=t,
                y=y,
                half_life=half_life,
                prev_eta=prev_eta,
            )

            prior_value = neg_logpdf_prior(
                theta=theta,
                sigma=sigma,
                phi=phi,
                p=p,
                prior_strength=prior_strength,
            )

            # Likelihood returns array on each data point, so we must sum it
            return np.sum(likelihood) + prior_value

        def posterior_gradient(
            theta, sigma, phi, p, *, tau, t, y, half_life=np.inf, prev_eta=None
        ) -> np.ndarray:
            r"""Return the gradient of the negative log posterior, which is the
            gradient of the negative log prior plus the gradient of the negative
            log likelihood. Should return array:
                [theta_1, theta_2, theta3, sigma, phi, p]
            The order of the gradient variables must match the order of the
            variables in the function signature."""

            likelihood_gradient = self.gradient(
                theta=theta,
                sigma=sigma,
                phi=phi,
                p=p,
                tau=tau,
                t=t,
                y=y,
                half_life=half_life,
                prev_eta=prev_eta,
            )
            prior_gradient = neg_logpdf_prior.gradient(
                theta=theta,
                sigma=sigma,
                phi=phi,
                p=p,
                prior_strength=prior_strength,
            )
            # Likelihood gradient has shape (num_parameters, num_datapoints)
            # and we must sum over every datapoint
            return np.sum(likelihood_gradient, axis=1) + prior_gradient

        parameter_names = ["theta", "sigma", "phi", "p"]
        fixed_params = {
            n: getattr(self, n) for n in parameter_names if getattr(self, n) is not None
        }

        # Here we bind all arguments that are fixed
        posterior = functools.partial(
            posterior,
            **fixed_params,
            tau=tau,
            t=t,
            y=y,
            half_life=half_life,
            prev_eta=None,
        )

        posterior_gradient = functools.partial(
            posterior_gradient,
            **fixed_params,
            tau=tau,
            t=t,
            y=y,
            half_life=half_life,
            prev_eta=None,
        )

        # Solve a linear regression problem to get an initial estimate
        # This works well, and is better than using theta prior as guess
        initial_params = self._initial_guess(t=t, y=y, tau=tau, half_life=np.inf)

        # Bounds values
        epsilon = 1e-6
        phi_lower, phi_upper = (epsilon, 1 - epsilon)
        p_lower, p_upper = (1 + epsilon, 2 - epsilon)

        def sigmoid_phi_derivative(x):
            y = sp.special.expit(x)
            return (phi_upper - phi_lower) * y * (1 - y)

        def sigmoid_p_derivative(x):
            y = sp.special.expit(x)
            return (p_upper - p_lower) * y * (1 - y)

        free_params = {
            "theta": Parameter(x0=np.array(initial_params["theta"]), bounds=(-30, 35)),
            "sigma": Parameter(
                x0=float(initial_params["sigma"]),
                bounds=(1e-3, None),
                transform=lambda x: np.exp(x) + epsilon,
                inverse_transform=lambda x: np.log(x - epsilon),
                transform_derivative=np.exp,
            ),
            "phi": Parameter(
                x0=0.66,
                bounds=(phi_lower, phi_upper),
                transform=lambda x: phi_lower
                + (phi_upper - phi_lower) * sp.special.expit(x),
                inverse_transform=lambda x: sp.special.logit(
                    (x - phi_lower) / (phi_upper - phi_lower)
                ),
                transform_derivative=sigmoid_phi_derivative,
            ),
            "p": Parameter(
                x0=1.5,
                bounds=(p_lower, p_upper),
                transform=lambda x: p_lower + (p_upper - p_lower) * sp.special.expit(x),
                inverse_transform=lambda x: sp.special.logit(
                    (x - p_lower) / (p_upper - p_lower)
                ),
                transform_derivative=sigmoid_p_derivative,
            ),
        }

        # If parameters are set to None, they are free and should be optimizer over
        free_params = {
            n: p for (n, p) in free_params.items() if getattr(self, n) is None
        }

        optimization_problem = Optimizer(
            posterior, grad=posterior_gradient, **free_params
        )

        # Extensive tests on hundreds of real-world wells show that other
        # optimizers never achieve lower objective func values than BFGS
        with np.errstate(all="warn"):
            opt_params, opt_result = optimization_problem.optimize_bfgs()
            assert np.isfinite(opt_result.fun), "Optimization func value not finite"

        return opt_params

    def gradient(
        self, theta, sigma, phi, p, *, tau, t, y, half_life=np.inf, prev_eta=None
    ):
        r"""Returns the gradient of the negative-log-likelihood with respect to:

        - theta
        - sigma
        - phi
        - p

        Returns an array of shape (len(theta) + 3, len(t))."""
        tau = np.array(tau, dtype=float)
        t = np.array(t, dtype=float)
        y = np.array(y, dtype=float)
        beta = p  # The API uses p, but the code uses beta. They are the same.

        # Prepare arrays for vectorized computations
        curve = self.curve_cls(*theta)
        log_f = curve.eval_log(t)
        log_y = np.log(y)

        # Setup variables
        log_y_minus_log_f = log_y - log_f
        log_y_minus_mu = (log_y_minus_log_f[1:]) - phi * (log_y_minus_log_f[:-1])

        # If the data match exactly, alpha will be zero and log(alpha) is -inf
        epsilon = np.finfo(float).eps  # 2.220446049250313e-16
        small_mask = np.abs(log_y_minus_mu) < epsilon
        log_y_minus_mu[small_mask] = epsilon * (
            2 * (log_y_minus_mu[small_mask] >= 0).astype(int) - 1
        )  # Sign

        # Compute log_y_minus_mu ** beta
        abs_pow_beta = np.exp(beta * np.log(np.abs(log_y_minus_mu)))

        sigma_sqrt_tau = sigma * np.sqrt(tau)

        # Compute sigma_sqrt_tau**beta
        sigma_sqrt_tau_pow_beta = np.exp(beta * np.log(np.abs(sigma_sqrt_tau)))

        partial_mu = (beta * abs_pow_beta) / (
            log_y_minus_mu * sigma_sqrt_tau_pow_beta[1:]
        )

        # Derivative wrt theta (curve parameters)
        curve_grad_log = curve.eval_grad_log(t)
        partial_theta = partial_mu * (
            curve_grad_log[:, 1:] - phi * curve_grad_log[:, :-1]
        )

        # Derivative with respect to sigma
        partial_sigma = (
            abs_pow_beta * beta / (sigma_sqrt_tau_pow_beta[1:] * sigma) - 1 / sigma
        )

        # Derivative wrt phi
        partial_phi = partial_mu * log_y_minus_log_f[:-1]

        # Derivative wrt beta
        z = np.abs(log_y_minus_mu) / sigma_sqrt_tau[1:]
        z_pow_beta = abs_pow_beta / sigma_sqrt_tau_pow_beta[1:]
        partial_beta = (
            1 / beta + sp.special.digamma(1 / beta) / beta**2 - z_pow_beta * np.log(z)
        )

        assert partial_theta.shape[1] == len(partial_sigma) == len(partial_phi)
        assert len(partial_sigma) == len(partial_beta)

        gradient = np.zeros(shape=(len(theta) + 3, len(t)))
        gradient[: len(theta), 1:] = partial_theta
        gradient[len(theta), 1:] = partial_sigma
        gradient[len(theta) + 1, 1:] = partial_phi
        gradient[len(theta) + 2, 1:] = partial_beta

        # Initial condition
        if prev_eta is None:
            sigma_0_sq = gennorm_std(scale=sigma, beta=beta) ** 2 / (1 - phi**2)

            partial_mu = (log_y[0] - log_f[0]) / sigma_0_sq
            gradient[: len(theta), 0] = partial_mu * curve_grad_log[:, 0]

            # sigma
            d_sigma_ish = (log_y_minus_log_f[0]) ** 2 / sigma_0_sq - 1
            gradient[len(theta), 0] = d_sigma_ish / sigma

            # phi
            gradient[len(theta) + 1, 0] = d_sigma_ish * phi / (1 - phi**2)

            # beta
            dgam_dbeta = (
                sp.special.digamma(1 / beta) - 3 * sp.special.digamma(3 / beta)
            ) / (2 * beta**2)
            gradient[len(theta) + 2, 0] = d_sigma_ish * dgam_dbeta

        else:
            # Setup variables
            alpha = log_y_minus_log_f[0] - phi * prev_eta
            partial_mu = (
                np.sign(alpha)
                * np.abs(alpha) ** (beta - 1)
                * beta
                / sigma_sqrt_tau_pow_beta[0]
            )

            # Derivative wrt theta (curve parameters)
            partial_theta = partial_mu * curve_grad_log[:, 0]

            # Derivative with respect to sigma
            partial_sigma = (
                beta * np.abs(alpha) ** beta / sigma_sqrt_tau_pow_beta[0] - 1
            ) / sigma

            # Derivative wrt phi
            partial_phi = partial_mu * prev_eta

            # Derivative wrt beta
            z = np.abs(alpha) / sigma_sqrt_tau[0]
            partial_beta = (
                1 / beta - z**beta * np.log(z) + sp.special.digamma(1 / beta) / beta**2
            )

            gradient[: len(theta), 0] = partial_theta
            gradient[len(theta), 0] = partial_sigma
            gradient[len(theta) + 1, 0] = partial_phi
            gradient[len(theta) + 2, 0] = partial_beta

        # TODO: Equation used to be tau ** (1.0 + beta / 2.0)
        # but then the weights are a function of beta and the gradient computation
        # becomes more involved.
        w = tau**2 * self.get_weights(t=t, tau=tau, half_life=half_life)
        w = (w / np.sum(w)) * np.sum(tau)

        # Elementwise, weighted negative log-likelihood
        return -(w * gradient)

    def negative_ll(
        self, theta, sigma, phi, p, *, tau, t, y, half_life=np.inf, prev_eta=None
    ):
        r"""Returns the elementwise negative log-likelihood of the model.
        
        Likelihood of :math:`\log(y_0)` if ``prev_eta=None``: 
        
        .. math::
        
            P(\log y_0) &\sim \mathcal{N}(\text{loc}, \text{scale}) \\
            \quad \text{loc} &= \log(f(t_i; \theta)) \\
            \quad \text{scale} &= \frac{\text{std}(\text{GN}(\text{std}=\sigma, \beta=p))}{\sqrt{1 - \phi^2}}
        
        Likelihood of :math:`\log(y_i)` given :math:`\log(y_{i-1})`:
        
        .. math::
        
            P(\log y_i \mid \log y_{i-1}) &\sim \text{GN}(\text{loc}, \text{scale}, \beta=p) \\
            \quad \text{loc} &= \log(f(t_i; \theta)) + \phi \left[ \log(y_{i-1}) - \log(f(t_{i-1}; \theta)) \right] \\
            \quad \text{scale} &= \sqrt{\tau_i} \sigma
        
        Returns a vector with entries :math:`-w_i \log P(\log y_i \mid \log y_{i-1})`.
        """
        tau = np.array(tau, dtype=float)
        t = np.array(t, dtype=float)
        y = np.array(y, dtype=float)
        assert np.all(tau > 0)
        assert len(tau) == len(t) == len(y)
        assert isinstance(sigma, numbers.Real) and sigma >= 0, f"{sigma=}"
        assert isinstance(phi, numbers.Real) and (0 <= phi <= 1)
        assert 1 <= p <= 2

        # Return high negative log-likelihood if numerical values are bad
        if (abs(phi - 1) < 1e-5) or abs(sigma) < 1e-8:  # mimic np.isclose, but faster
            return np.inf

        # Prepare arrays for vectorized computations
        curve = self.curve_cls(*theta)
        log_f = curve.eval_log(t)
        log_y = np.log(y)
        ll = np.zeros_like(y, dtype=float)

        # Boundary condition (first term)
        if prev_eta is None:  # Long running behavior of AR(1) process
            loc = log_f[0]
            scale = gennorm_std(beta=p, scale=sigma) / np.sqrt(1 - phi**2)
            # gennorm_logpdf(x, loc, beta=2, scale=scale * sqrt(2)) = norm.logpdf
            ll[0] = gennorm_logpdf(
                x=log_y[0], beta=2.0, loc=loc, scale=scale * np.sqrt(2.0)
            )
        else:  # If an exact value of prev_eta is known, use it
            loc = log_f[0] + phi * prev_eta
            scale = np.sqrt(tau[0]) * sigma
            ll[0] = gennorm_logpdf(x=log_y[0], beta=p, loc=loc, scale=scale)

        # Remaining terms
        loc = log_f[1:] + phi * (log_y[:-1] - log_f[:-1])
        scale = np.sqrt(tau[1:]) * sigma
        ll[1:] = gennorm_logpdf(x=log_y[1:], beta=p, loc=loc, scale=scale)

        # Weights based on half-life (to weight more recent data more heavily)
        # and tau. We weight by tau to undo some of the heavy influence placed
        # on observations with low tau values (low tau -> low variance -> high
        # influence). Since the generalized normal has sqrt(tau)^p in the
        # denominator in log-space, we multiply by p/2 to undo the scaling,
        # then by sqrt(p) again to down weight
        w = tau ** (1.0 + 2.0 / 2.0) * self.get_weights(
            t=t, tau=tau, half_life=half_life
        )
        w = (w / np.sum(w)) * np.sum(tau)

        # Elementwise, weighted negative log-likelihood
        return -(w * ll)

    @staticmethod
    def get_weights(*, t, tau, half_life=np.inf):
        """Return weights for exponential decay going back in time. If half_life
        is set to infinity, then all weights become unity.

        Examples
        --------
        >>> tau = np.ones(4)  # Assuming tau is constant for simplicity
        >>> t = np.cumsum(tau) - tau / 2
        >>> AR1Model.get_weights(t=t, tau=tau)
        array([1., 1., 1., 1.])
        >>> AR1Model.get_weights(t=t, tau=tau, half_life=2)
        array([0.552..., 0.781..., 1.104... , 1.562...])
        """
        # Weights based on half-life
        time_ago = (t[-1] + tau[-1] / 2.0) - t  # Count periods back in time

        # As half-life becomes infinite, the weights become equal
        if np.isinf(half_life):
            w = np.ones_like(time_ago)
        else:
            w = np.power(2.0, -time_ago / half_life)

        return (w / np.sum(w)) * np.sum(tau)  # Normalize to sum(tau)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
