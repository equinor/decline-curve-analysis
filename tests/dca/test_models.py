import numpy as np
import pytest
import scipy as sp

from dca.decline_curve_analysis import Constant, Exponential
from dca.models import AR1Model, Prior


@pytest.mark.parametrize("seed", range(99))
def test_default_prior(seed):
    """Test that the gradient of the default prior works."""

    rng = np.random.default_rng(seed)
    prior_strength = 0.333

    default_prior = Prior(theta_mean=np.zeros(3))

    def func(x0):
        """Function with the signature required by `check_grad`"""
        *theta, sigma, phi, p = x0
        theta = tuple(theta)
        return default_prior(
            theta=theta, sigma=sigma, phi=phi, p=p, prior_strength=prior_strength
        )

    def grad(x0):
        """Function with the signature required by `check_grad`"""
        *theta, sigma, phi, p = x0
        theta = tuple(theta)
        return default_prior.gradient(
            theta=theta, sigma=sigma, phi=phi, p=p, prior_strength=prior_strength
        )

    x0 = np.array([1, 2, 3, 0.5, 0.5, 1.5])
    epsilon = (
        rng.random(size=len(x0)) - 0.5
    ) * 0.9  # rng.normal(size=len(x0), scale=0.1)

    # Check RMSE of gradient
    rmse = sp.optimize.check_grad(func=func, grad=grad, x0=x0 + epsilon, seed=rng)

    assert np.isclose(rmse, 0.0, atol=1e-3)


class TestAR1Model:
    def test_on_concrete_example(self):
        # Test that we do not regress on these results, which were obtained
        # by running 3 optimizers for 1000 iterations and getting the best
        # values. With gradient, and with fewer iterations, we should still
        # be close.

        rng = np.random.default_rng(42)

        n = 100
        tau = rng.random(n) * 0.5 + 0.5
        t = np.cumsum(tau) - tau / 2

        theta = (6, 4)
        sigma = 0.2
        phi = 0.1
        p = 1.5

        # Generate data
        model = AR1Model(curve_cls=Exponential)
        model.update(sigma=sigma, phi=phi, theta=theta, p=p)
        y = model.simulate(tau=tau, t=t, seed=rng, simulations=1)[0, :]

        # Optimize parameters
        model.update(sigma=None, phi=None, theta=None, p=None)
        result = model.optimize(tau=tau, t=t, y=y, prior_strength=0.33)

        # Evaluate at optimum
        assert (
            model.negative_ll(**result, tau=tau, t=t, y=y).sum()
            <= -27.467371795471322 + 1e-8
        )

    @pytest.mark.parametrize("seed", range(99))
    @pytest.mark.parametrize("prev_eta", [-0.8294, None, 0.666])
    def test_gradient_numerical_vs_analytical(self, seed, prev_eta):
        # -------------------------
        rng = np.random.default_rng(seed)

        # Use relatively few data points, for speed and for initial condition
        n = 3
        tau = rng.random(n) * 0.5 + 0.5
        t = np.cumsum(tau) - tau / 2

        # Parameters for simulation
        curve_params = (
            10 + rng.normal(scale=0.5),
            0.002 + rng.normal(scale=0.002 / 10),
        )
        curve_func = Exponential.from_original_parametrization(*curve_params)
        theta = curve_func.thetas
        sigma = 0.2 + rng.normal(scale=0.05)
        phi = rng.beta(a=3, b=2)  # expected value 0.66
        p = 1 + rng.beta(a=2, b=2)  # expected value 0.5

        # Create a model and simulate from it
        model = AR1Model(curve_cls=Exponential)
        model.update(theta=theta, sigma=sigma, phi=phi, p=p)
        y = model.simulate(tau=tau, t=t, seed=rng, simulations=1)[0, :]

        def func(x0):
            """Function with the signature required by `check_grad`"""
            *theta, sigma, phi, p = x0
            theta = tuple(theta)
            kwargs = {"tau": tau, "t": t, "y": y, "prev_eta": prev_eta}
            return model.negative_ll(
                theta=theta, sigma=sigma, phi=phi, p=p, **kwargs
            ).sum()

        def grad(x0):
            """Function with the signature required by `check_grad`"""
            *theta, sigma, phi, p = x0
            theta = tuple(theta)
            kwargs = {"tau": tau, "t": t, "y": y, "prev_eta": prev_eta}
            return model.gradient(theta=theta, sigma=sigma, phi=phi, p=p, **kwargs).sum(
                axis=1
            )

        x0 = np.array([8, 6, 0.5, 0.5, 1.5])
        epsilon = (rng.random(size=len(x0)) - 0.5) * 0.8

        # Check RMSE of gradient
        rmse = sp.optimize.check_grad(func=func, grad=grad, x0=x0 + epsilon, seed=rng)

        assert np.isclose(rmse, 0.0, atol=1e-3)

    def test_that_constant_model_std_is_inferred(self):
        std = 1.0

        rng = np.random.default_rng(42)

        # The simple gaussian case with no autocorrelation
        tau = np.ones(1000)
        t = np.cumsum(tau) - tau / 2
        y = np.exp(rng.normal(loc=0, scale=std, size=1000))

        # Infer std naively
        naive_est_std = np.std(np.log(y))
        assert np.isclose(std, naive_est_std, atol=0.02)

        # Infer std using model
        model = AR1Model(curve_cls=Constant)
        model.update(theta=None, sigma=None, phi=0.001, p=1.999)

        opt_params = model.optimize(tau=tau, t=t, y=y)
        est_std = model.pointwise_std(p=2, sigma=opt_params["sigma"], phi=0)

        assert np.isclose(est_std, naive_est_std, atol=0.001)

    def test_that_low_time_on_matters_little(self):
        # The simple gaussian case with no autocorrelation
        model = AR1Model(curve_cls=Constant)
        model.update(theta=None, sigma=1, phi=0, p=2)

        tau = np.array([1, 1])
        t = np.cumsum(tau) - tau / 2
        y = np.array([2, 2])
        opt_params1 = model.optimize(tau=tau, t=t, y=y)
        assert np.isclose(opt_params1["theta"], np.log(2))

        # Add a little outlier observed for a short amount of time
        # Since longer time on means more uncertainty, this observation obtains
        # very high uncertainty. This means the mean value is influenced by it
        # quite a lot.
        # The model uses likelihood weightig to counter this effect.
        # The net result is that the final data point matters very little
        tau = np.array([1, 1, 1e-3])
        t = np.cumsum(tau) - tau / 2
        y = np.array([2, 2, 5])
        # np.mean(np.log([2, 2, 5])) => 0.99857...
        # np.mean(np.log([2, 2]))    => 0.69314...

        # I would assume that THETA should be between 0.69314 and 0.99857
        # Depending on how exactly the likelihood weighting balances the
        # inverse variance weighting and effect of time on uncertainty.
        opt_params2 = model.optimize(tau=tau, t=t, y=y)
        assert np.isclose(opt_params2["theta"], 0.6936051)

    def test_estimation_at_different_resolutions(self):
        # The simple gaussian case with no autocorrelation
        model = AR1Model(curve_cls=Constant)
        model.update(theta=None, sigma=1, phi=0, p=2)

        # Resolution frequency=1 over two periods => 2 observations
        tau = np.array([1, 1])
        t = np.cumsum(tau) - tau / 2
        y = np.array([2, 3])
        opt_params1 = model.optimize(tau=tau, t=t, y=y)

        # Resolution frequency=0.1 over two periods => 20 observations
        tau = np.array([0.1] * 20)
        t = np.cumsum(tau) - tau / 2
        y = np.array([2] * 10 + [3] * 10)
        opt_params2 = model.optimize(tau=tau, t=t, y=y)

        assert np.isclose(opt_params1["theta"], opt_params2["theta"], atol=0.01)

    def test_std_estimation_at_different_resolutions(self):
        rng = np.random.default_rng(42)

        # The simple gaussian case with no autocorrelation
        model = AR1Model(curve_cls=Constant)
        model.update(theta=None, sigma=None, phi=0.001, p=1.999)

        tau = np.array([1] * 1000)
        t = np.cumsum(tau) - tau / 2
        y = np.exp(rng.normal(scale=np.sqrt(tau)))
        opt_params1 = model.optimize(tau=tau, t=t, y=y)

        tau = np.array([0.5] * 2000)
        t = np.cumsum(tau) - tau / 2
        y = np.exp(rng.normal(scale=np.sqrt(tau)))
        opt_params2 = model.optimize(tau=tau, t=t, y=y)

        assert np.isclose(opt_params1["sigma"], opt_params2["sigma"], rtol=0.05)

    @pytest.mark.parametrize("seed", range(25))
    def test_parameter_estimation(self, seed):
        """Create a synthetic data set using know parameters.
        Then estimate the parameters with optimization.
        Check that the estimated parameters lead to lower negative log likelihood
        compared to the parameters used to generate the data set."""
        rng = np.random.default_rng(seed)

        # Parameters for simulation
        curve_params = (
            10 + rng.normal(scale=0.5),
            0.002 + rng.normal(scale=0.002 / 10),
        )
        curve_func = Exponential.from_original_parametrization(*curve_params)
        theta = curve_func.thetas
        sigma = 0.2 + rng.normal(scale=0.05)
        phi = rng.beta(a=3, b=2)  # expected value 0.66
        p = 1 + rng.beta(a=2, b=2)  # expected value 0.5

        # Create a model and simulate from it
        model = AR1Model(curve_cls=Exponential)
        model.update(sigma=sigma, phi=phi, theta=theta, p=p)

        # Simulations
        n = 2 ** rng.integers(low=6, high=10)  # between
        tau = rng.triangular(left=0.6, mode=1.0, right=1.0, size=n)
        # tau = np.ones(n)  # Assuming tau is constant for simplicity
        t = np.cumsum(tau) - tau / 2
        y = model.simulate(tau=tau, t=t, seed=3, simulations=1)[0, :]

        class DummyPrior:
            def __call__(self, theta, sigma, phi, p, prior_strength):
                return 0

            def gradient(self, theta, sigma, phi, p, prior_strength):
                return np.zeros(len(theta) + 3)

        neg_logpdf_prior = DummyPrior()

        # Set parameter as free (indicated by using None)
        model.update(sigma=None, phi=None, theta=None, p=None)

        opt_params = model.optimize(
            tau=tau,
            t=t,
            y=y,
            prior_strength=0.0,
            half_life=np.inf,
            neg_logpdf_prior=neg_logpdf_prior,
        )

        # Negative log-likelihood at estimated params should be lower
        neg_ll_opt = model.negative_ll(**opt_params, tau=tau, t=t, y=y).sum()
        sim_params = {"theta": theta, "sigma": sigma, "phi": phi, "p": p}
        neg_ll_sim = model.negative_ll(**sim_params, tau=tau, t=t, y=y).sum()
        assert neg_ll_opt < neg_ll_sim


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
            "--capture=sys",
            # "-x",
            # "-k test_derivative_wrt_theta",
        ]
    )
