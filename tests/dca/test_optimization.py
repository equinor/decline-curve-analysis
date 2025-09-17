import functools

import numpy as np
import pytest
import scipy as sp

from dca.optimization import Optimizer, Parameter


class LinearProblem:
    """A linear regression problem, used for testing."""

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def loss(self, beta, sigma, k=2):
        """Negative log-pdf of the normal distribution."""
        A, b = self.A, self.b
        mu = A @ beta
        return -sp.stats.norm(loc=mu, scale=sigma).logpdf(b).sum()

    def gradient(self, beta, sigma):
        """Gradient, must return a numpy array."""
        A, b = self.A, self.b

        # Gradient wrt beta
        mu = A @ beta
        grad_beta = (mu - b) @ A / sigma**2

        # Gradient wrt sigma
        grad_sigma = (1 / sigma - (mu - b) ** 2 / sigma**3).sum()

        gradient = np.zeros(len(beta) + 1)
        gradient[:-1] = grad_beta
        gradient[-1] = grad_sigma
        return gradient


class TestOptimizer:
    def test_gradient_on_linear_problem(self):
        # Problem data
        rng = np.random.default_rng(42)
        A = rng.normal(size=(10, 2))
        b = rng.normal(size=10)

        # Problem instance
        problem = LinearProblem(A=A, b=b)

        # Gradient check
        def func(x0):
            *beta, sigma = x0
            return problem.loss(beta=beta, sigma=sigma)

        def grad(x0):
            *beta, sigma = x0
            return problem.gradient(beta=beta, sigma=sigma)

        x0 = np.array([1, 2, 1])
        rmse = sp.optimize.check_grad(func, grad, x0=x0)
        assert rmse < 1e-5

    @pytest.mark.parametrize("use_gradient", [True, False])
    def test_lbfgsb_on_linear_problem(self, use_gradient):
        # Problem data
        rng = np.random.default_rng(42)
        A = rng.normal(size=(10, 2))
        b = rng.normal(size=10)

        # Problem instance
        problem = LinearProblem(A=A, b=b)

        # Bind a parameter, then optimize
        loss = functools.partial(problem.loss, sigma=1.0)
        grad = functools.partial(problem.gradient, sigma=1.0)
        optimization_problem = Optimizer(
            loss=loss,
            grad=grad if use_gradient else None,
            beta=Parameter(x0=np.array([1, 2])),
        )

        opt_params, opt_result = optimization_problem.optimize_lbfgsb()

        assert np.allclose(opt_params["beta"], np.array([-0.33824323, -0.13242369]))

        # Optimize with no bound parameters
        optimization_problem = Optimizer(
            loss=problem.loss,
            grad=problem.gradient if use_gradient else None,
            beta=Parameter(x0=np.array([1, 2])),
            sigma=Parameter(x0=3.0, bounds=(1e-6, None)),
        )

        opt_params, _opt_result = optimization_problem.optimize_lbfgsb()

        assert np.allclose(opt_params["beta"], np.array([-0.33824323, -0.13242369]))
        assert np.isclose(opt_params["sigma"], 0.4437543289665066)

    @pytest.mark.parametrize("use_gradient", [True, False])
    def test_bfgs_on_linear_problem(self, use_gradient):
        # Problem data
        rng = np.random.default_rng(42)
        A = rng.normal(size=(10, 2))
        b = rng.normal(size=10)

        # Problem instance
        problem = LinearProblem(A=A, b=b)

        # Bind a parameter, then optimize
        beta = np.array([-0.33824323, -0.13242369])
        loss = functools.partial(problem.loss, beta=beta)
        grad = functools.partial(problem.gradient, beta=beta)
        optimization_problem = Optimizer(
            loss=loss,
            grad=grad if use_gradient else None,
            sigma=Parameter(
                x0=3.0,
                transform=np.exp,
                inverse_transform=np.log,
                transform_derivative=np.exp,
            ),
        )

        opt_params, opt_result = optimization_problem.optimize_bfgs()

        assert np.isclose(opt_params["sigma"], 0.4437543289665066)

        def sigmoid(x):
            """Map from R to (-5, 5)."""
            return sp.special.expit(x) * 10 - 5

        def sigmoid_inv(x):
            """Map from (-5, 5) to R."""
            return sp.special.logit((x + 5) / 10)

        def sigmoid_derivative(x):
            y = sp.special.expit(x)
            return 10 * y * (1 - y)

        # Optimize with no bound parameters
        optimization_problem = Optimizer(
            loss=problem.loss,
            grad=problem.gradient if use_gradient else None,
            beta=Parameter(
                x0=np.array([1, 2]),
                transform=sigmoid,
                inverse_transform=sigmoid_inv,
                transform_derivative=sigmoid_derivative,
            ),
            sigma=Parameter(
                x0=3.0,
                transform=np.exp,
                inverse_transform=np.log,
                transform_derivative=np.exp,
            ),
        )

        opt_params, _opt_result = optimization_problem.optimize_bfgs()

        assert np.allclose(opt_params["beta"], np.array([-0.33824323, -0.13242369]))
        assert np.isclose(opt_params["sigma"], 0.4437543289665066)

    @pytest.mark.parametrize("use_gradient", [True, False])
    def test_nelder_mead_on_linear_problem(self, use_gradient):
        # Problem data
        rng = np.random.default_rng(42)
        A = rng.normal(size=(10, 2))
        b = rng.normal(size=10)

        # Problem instance
        problem = LinearProblem(A=A, b=b)

        # Bind a parameter, then optimize
        beta = np.array([-0.33824323, -0.13242369])
        loss = functools.partial(problem.loss, beta=beta)
        grad = functools.partial(problem.gradient, beta=beta)
        optimization_problem = Optimizer(
            loss=loss,
            grad=grad if use_gradient else None,
            sigma=Parameter(
                x0=3.0,
                transform=np.exp,
                inverse_transform=np.log,
                transform_derivative=np.exp,
            ),
        )

        opt_params, opt_result = optimization_problem.optimize_nelder_mead()

        assert np.isclose(opt_params["sigma"], 0.4437543289665066)

        def sigmoid(x):
            """Map from R to (-5, 5)."""
            return sp.special.expit(x) * 10 - 5

        def sigmoid_inv(x):
            """Map from (-5, 5) to R."""
            return sp.special.logit((x + 5) / 10)

        def sigmoid_derivative(x):
            y = sp.special.expit(x)
            return 10 * y * (1 - y)

        # Optimize with no bound parameters
        optimization_problem = Optimizer(
            loss=problem.loss,
            grad=problem.gradient if use_gradient else None,
            beta=Parameter(
                x0=np.array([1, 2]),
                transform=sigmoid,
                inverse_transform=sigmoid_inv,
                transform_derivative=sigmoid_derivative,
            ),
            sigma=Parameter(
                x0=3.0,
                transform=np.exp,
                inverse_transform=np.log,
                transform_derivative=np.exp,
            ),
        )

        opt_params, _opt_result = optimization_problem.optimize_nelder_mead()

        assert np.allclose(opt_params["beta"], np.array([-0.33820371, -0.13249737]))
        assert np.isclose(opt_params["sigma"], 0.44379245748549123)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys", "-x"])
