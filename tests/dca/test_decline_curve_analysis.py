"""
Test the DCA code.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize

from dca.datasets import load_monthly_sodir_production
from dca.decline_curve_analysis import (
    Arps,
    CurveLoss,
    Exponential,
    log1pexp,
    softplus,
    softplus_inv,
)
from dca.models import AR1Model
from dca.timeseries import preprocess_timeseries, to_producing_time

np.random.seed(1)


THETAS = [
    (
        np.random.normal(loc=1.1, scale=0.1),
        np.random.normal(loc=0.9, scale=0.1),
        np.random.normal(loc=-2, scale=0.1),
    )
    for _ in range(99)
]

TOLERANCE = 1e-8

PRODUCTION_AND_DAYS_ON = [
    ([30, 0.5, 30], [30, 3, 30]),
    ([30, 0.5, 1], [5, 3, 1]),
    ([30, 0.5, 300], [1, 3, 1]),
    ([1, 0.5, 30], [5, 3, 20]),
    ([3, 1, 0.5], [1, 30, 1]),
    ([100, 1, 200], [1, 30, 1]),
]

# Get some fields to test on
FIELD_NAMES = [
    "BALDER",
    "DRAUGEN",
    "GOLIAT",
    "GULLFAKS",
    "GYDA",
    "HEIDRUN",
    "JOTUN",
    "KRISTIN",
    "NORNE",
    "OSEBERG",
    "OSEBERG SØR",
    "STATFJORD",
    "STATFJORD ØST",
    "TYRIHANS",
    "ULA",
    "VESLEFRIKK",
]


class TestOnRealData:
    """Test on real-world SODIR data. This data is on the field level.
    Since a field consists of many wells, the assumptions of DCA are probably
    violated a bit. Production for a field is rarely declining like a DCA
    curve assumes, since new wells are drilled and production can increase
    as time passes."""

    @pytest.mark.parametrize("field_name", FIELD_NAMES)
    @pytest.mark.parametrize("curve_cls", [Arps, Exponential])
    def test_ar1model_on_SODIR_data(self, field_name, curve_cls):
        # Load data and filter it
        df = load_monthly_sodir_production()
        df_field = df[df["prfInformationCarrier"] == field_name]
        df_field = df_field.sort_values(["prfYear", "prfMonth"])

        # Preprocess
        time_on, production, mask = preprocess_timeseries(
            time_on=np.ones(len(df_field)),
            production=df_field["prfPrdOilNetMillSm3"].values,
        )

        # Impute values, then mask them
        df_field = df_field.assign(prfPrdOilNetMillSm3=lambda df: production)
        df_field = df_field.loc[mask, :]
        if df_field.empty:
            return

        # Prepare for curve fitting. y = production / month, x = months
        x, y, w = to_producing_time(
            np.ones(len(df_field)), df_field["prfPrdOilNetMillSm3"].values
        )

        # Set up model and optimize it
        model = AR1Model(curve_cls=curve_cls)
        opt_params = model.optimize(tau=w, t=x, y=y, prior_strength=1e-2, half_life=100)

        # Simulate using parameters
        model.update(**opt_params)
        seed = int.from_bytes(field_name.encode("utf-8"), byteorder="big") % (2**10)
        y_sims = model.simulate(tau=w, t=x, seed=seed, prev_eta=None, simulations=99)
        y_sim = np.mean(y_sims, axis=0)

        # plt.plot(x, y_sim); plt.plot(x, y);
        mae_model = np.median(np.abs(y_sim[-200:] - y[-200:]))
        assert mae_model < 0.075

        # On fields with some data, check that we at not much worse than median
        mae_const = np.median(np.abs(np.median(y[-200:]) - y[-200:]))
        assert mae_model < mae_const * 2.1


class TestExponential:
    @pytest.mark.parametrize("theta", THETAS)
    def test_conversion_from_parametrization_and_back(self, theta):
        theta = theta[:2]
        params_original = Exponential(*theta).original_parametrization()
        exp_model = Exponential.from_original_parametrization(*params_original)

        assert np.allclose(exp_model.thetas, theta)

    def test_conversion_from_original_parametrization_and_back(self):
        C = 10
        k = 0.1
        exp_model = Exponential.from_original_parametrization(C, k)

        C_back, k_back = exp_model.original_parametrization()

        assert np.allclose([C, k], [C_back, k_back])

    @pytest.mark.parametrize("factor", [0.33, 0.5, 0.8, 1.2, 2])
    def test_that_theta_1_is_exp_of_integral(self, factor):
        # Create a base model
        theta_1 = 5

        model = Exponential(theta_1, 0.66)
        integral_1, _ = quad(lambda t: np.exp(model.eval_log(t)), 0, 100 * 365)

        # We want to scale the EUR (integral from 0 to infty) by a factor,
        # and this corresponds to adding log(factor) to theta_1:
        scaled_theta_1 = np.log(factor) + theta_1
        scaled_model = Exponential(scaled_theta_1, 0.66)
        with np.errstate(under="ignore"):
            integral_2, _ = quad(
                lambda t: np.exp(scaled_model.eval_log(t)), 0, 100 * 365
            )

        assert np.isclose(factor, integral_2 / integral_1)

    @pytest.mark.parametrize("theta", THETAS)
    @pytest.mark.parametrize("t", [1, 2.0, np.array([1.0, 3.0])])
    def test_gradient(self, theta, t):
        theta = theta[:2]  # Exponential uses two parameters

        # Compute analytical gradient
        grad_analytical = Exponential(*theta).eval_grad_log(t)

        # Compute numerical gradient
        epsilon = 1e-8

        # Add epsilon to each parameter individually
        # https://en.wikipedia.org/wiki/Numerical_differentiation#Finite_differences
        grad_numerical = []
        for i, _ in enumerate(theta):
            theta_plus_eps = list(theta)
            theta_plus_eps[i] = theta_plus_eps[i] + epsilon

            theta_minus_eps = list(theta)
            theta_minus_eps[i] = theta_minus_eps[i] - epsilon

            g = (
                Exponential(*theta_plus_eps).eval_log(t)
                - Exponential(*theta_minus_eps).eval_log(t)
            ) / (2 * epsilon)
            grad_numerical.append(g)

        grad_numerical = np.array(grad_numerical)
        assert np.allclose(grad_analytical, grad_numerical, atol=1e-4)

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral(self, theta):
        theta = theta[:2]  # Exponential uses two parameters

        a = 1
        b = 3

        # Perform numerical integration
        result, _ = quad(lambda t: np.exp(Exponential(*theta).eval_log(t)), a, b)

        # Use analytical results for integration
        analytical_result = Exponential(*theta).eval_integral(b) - Exponential(
            *theta
        ).eval_integral(a)

        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_squared_integral(self, theta):
        theta = theta[:2]  # Exponential uses two parameters

        a = 1
        b = 3

        # Perform numerical integration
        result, _ = quad(lambda t: np.exp(Exponential(*theta).eval_log(t)) ** 2, a, b)

        # Use analytical results for integration
        analytical_result = Exponential(*theta).eval_squared_integral(b) - Exponential(
            *theta
        ).eval_squared_integral(a)

        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral_from_0_to_inf(self, theta):
        theta = theta[:2]  # Exponential uses two parameters

        a, b = 0, 999

        with np.errstate(under="ignore"):
            result, _ = quad(lambda t: np.exp(Exponential(*theta).eval_log(t)), a, b)
        analytical_result = Exponential(*theta).eval_integral_from_0_to_inf()
        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral_from_0_to_T(self, theta):
        theta = theta[:2]  # Exponential uses two parameters

        a, b = 0, 13.3

        result, _ = quad(lambda t: np.exp(Exponential(*theta).eval_log(t)), a, b)
        analytical_result = Exponential(*theta).eval_integral_from_0_to_T(b)
        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral_from_T_to_inf(self, theta):
        theta = theta[:2]  # Exponential uses two parameters

        a, b = 3, 999

        with np.errstate(under="ignore"):
            result, _ = quad(lambda t: np.exp(Exponential(*theta).eval_log(t)), a, b)
        analytical_result = Exponential(*theta).eval_integral_from_T_to_inf(a)
        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("scale", [0.5, 1, 2])
    @pytest.mark.parametrize("shift", [0, 1, 2])
    @pytest.mark.parametrize("theta", THETAS[:10])
    def test_scaling_and_shifting(self, theta, scale, shift):
        theta = theta[:2]  # Exponential uses two parameters
        model = Exponential(*theta)
        x_grid = np.linspace(0, 50)

        # Evaluate on original grid, then check that
        # transforming model = transforming grid
        eval_original = model(x_grid)
        model_transformed = model.shift_and_scale(shift=shift, scale=scale)
        eval_transformed = model_transformed(x_grid * scale + shift)

        assert np.allclose(eval_original, eval_transformed)


class TestArps:
    @pytest.mark.parametrize("factor", [0.33, 0.5, 0.8, 1.2, 2])
    def test_that_theta_1_is_exp_of_integral(self, factor):
        # Create a base model
        theta_1 = 5

        model = Arps(theta_1, 7, 1)
        integral_1, _ = quad(lambda t: np.exp(model.eval_log(t)), 0, 100 * 365)

        # We want to scale the EUR (integral from 0 to infty) by a factor,
        # and this corresponds to adding log(factor) to theta_1:
        scaled_theta_1 = np.log(factor) + theta_1
        scaled_model = Arps(scaled_theta_1, 7, 1)
        integral_2, _ = quad(lambda t: np.exp(scaled_model.eval_log(t)), 0, 100 * 365)

        assert np.isclose(factor, integral_2 / integral_1)

    @pytest.mark.parametrize("theta", THETAS)
    @pytest.mark.parametrize("t", [1, 2.0, np.array([1.0, 3.0])])
    def test_gradient(self, theta, t):
        # Compute analytical gradient
        grad_analytical = Arps(*theta).eval_grad_log(t)

        # Compute numerical gradient
        epsilon = 1e-8

        # Add epsilon to each parameter individually
        # https://en.wikipedia.org/wiki/Numerical_differentiation#Finite_differences
        grad_numerical = []
        for i, _ in enumerate(theta):
            theta_plus_eps = list(theta)
            theta_plus_eps[i] = theta_plus_eps[i] + epsilon

            theta_minus_eps = list(theta)
            theta_minus_eps[i] = theta_minus_eps[i] - epsilon

            g = (
                Arps(*theta_plus_eps).eval_log(t) - Arps(*theta_minus_eps).eval_log(t)
            ) / (2 * epsilon)
            grad_numerical.append(g)

        grad_numerical = np.array(grad_numerical)
        assert np.allclose(grad_analytical, grad_numerical, atol=1e-4)

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral(self, theta):
        a = 1
        b = 3

        # Perform numerical integration
        result, _ = quad(lambda t: np.exp(Arps(*theta).eval_log(t)), a, b)

        # Use analytical results for integration
        analytical_result = Arps(*theta).eval_integral(b) - Arps(*theta).eval_integral(
            a
        )

        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_squared_integral(self, theta):
        a = 1
        b = 3

        # Perform numerical integration
        result, _ = quad(lambda t: np.exp(Arps(*theta).eval_log(t)) ** 2, a, b)

        # Use analytical results for integration
        analytical_result = Arps(*theta).eval_squared_integral(b) - Arps(
            *theta
        ).eval_squared_integral(a)

        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral_from_0_to_inf(self, theta):
        a, b = 0, 999

        result, _ = quad(lambda t: np.exp(Arps(*theta).eval_log(t)), a, b)
        analytical_result = Arps(*theta).eval_integral_from_0_to_inf()
        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral_from_0_to_T(self, theta):
        a, b = 0, 13.3

        result, _ = quad(lambda t: np.exp(Arps(*theta).eval_log(t)), a, b)
        analytical_result = Arps(*theta).eval_integral_from_0_to_T(b)
        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_integral_from_T_to_inf(self, theta):
        a, b = 3, 999

        result, _ = quad(lambda t: np.exp(Arps(*theta).eval_log(t)), a, b)
        analytical_result = Arps(*theta).eval_integral_from_T_to_inf(a)
        assert abs(result - analytical_result) < TOLERANCE

    @pytest.mark.parametrize("theta", THETAS)
    def test_that_log_original_exponentiated_equals_evaluation(self, theta):
        t = np.arange(100) + 1
        params_original = Arps(*theta).original_parametrization()

        log_exp_evaluated = np.exp(
            Arps.eval_log_original_parametrization(t, *params_original)
        )
        evaluated = Arps.eval_original_parametrization(t, *params_original)

        assert np.allclose(log_exp_evaluated, evaluated)

    @pytest.mark.parametrize("theta", THETAS)
    def test_that_original_parametrization_equals_Lees_parametrization(self, theta):
        # Grid to evaluate on
        x_grid = np.arange(100) + 1

        # Evaluate using Lee's parametrization from the paper
        eval_theta_parametrization = Arps(*theta).eval_log(x_grid)

        # Get original parameters
        params_original = Arps(*theta).original_parametrization()
        eval_arps_parametrization = Arps.eval_log_original_parametrization(
            x_grid, *params_original
        )

        # Answer should be the same
        assert np.allclose(eval_theta_parametrization, eval_arps_parametrization)

    @pytest.mark.parametrize("theta", THETAS)
    def test_conversion_from_Lees_parametrization_and_back(self, theta):
        # Get Arps parameters
        params_original = Arps(*theta).original_parametrization()

        # Convert to theta representation
        arp_model = Arps.from_original_parametrization(*params_original)

        assert np.allclose(arp_model.thetas, theta)

    def test_conversion_from_original_parametrization_and_back(self):
        q_1 = 10
        h = 0.5
        D = 0.5
        # Convert to theta representation
        arp_model = Arps.from_original_parametrization(q_1, h, D)

        q_1_back, h_back, D_back = arp_model.original_parametrization()

        assert np.allclose([q_1, h, D], [q_1_back, h_back, D_back])

    @pytest.mark.parametrize("scale", [0.5, 1, 2])
    @pytest.mark.parametrize("shift", [0, 1, 2])
    @pytest.mark.parametrize("theta", THETAS[:10])
    def test_scaling_and_shifting(self, theta, scale, shift):
        model = Arps(*theta)
        x_grid = np.linspace(0, 50)

        # Evaluate on original grid, then check that
        # transforming model = transforming grid
        eval_original = model(x_grid)
        model_transformed = model.shift_and_scale(shift=shift, scale=scale)
        eval_transformed = model_transformed(x_grid * scale + shift)

        assert np.allclose(eval_original, eval_transformed)


# %%
class TestSoftPlus:
    def test_that_numerically_stable_equals_naive(self):
        # Test that softplus with numerical stability equals normal softplus
        assert np.isclose(softplus(0), np.log1p(np.exp(0)))
        assert np.isclose(softplus(1), np.log1p(np.exp(1)))
        assert np.isclose(softplus(-3), np.log1p(np.exp(-3)))
        assert np.isclose(softplus(5.5), np.log1p(np.exp(5.5)))
        assert np.isclose(softplus(10), np.log1p(np.exp(10)))

    def test_that_numerically_stable_inverse_equals_naive(self):
        # Test that inverse softplus with numerical stability
        # equals the naive implementation of inverse softplus
        assert np.isclose(softplus_inv(1), np.log(np.exp(1) - 1))
        assert np.isclose(softplus_inv(2), np.log(np.exp(2) - 1))
        assert np.isclose(softplus_inv(3), np.log(np.exp(3) - 1))

    def test_that_inverse_maps_back(self):
        # Test that functions are inverses
        for x in [-3, -2, 0, 2, 3, 5, 10**2, 10**5]:
            assert np.isclose(softplus_inv(softplus(x)), x)
            if x > 0:
                assert np.isclose(softplus(softplus_inv(x)), x)


class TestFullCurveFittingExamplesWithExponential:
    @pytest.mark.parametrize("theta", THETAS)
    def test_that_curve_fitting_with_gradient_works(self, theta):
        theta = theta[:2]  # Exponential only uses two

        # Create a curve
        t = np.linspace(0, 25, num=2**7)
        y = Exponential(*theta).eval(t)

        # Learn parameters
        def log_Exponential(t, theta1, theta2):
            return Exponential(theta1, theta2).eval_log(t)

        def log_Exponential_grad(t, theta1, theta2):
            return Exponential(theta1, theta2).eval_grad_log(t)

        loss = CurveLoss(
            curve_func=log_Exponential, p=2.0, curve_func_grad=log_Exponential_grad
        )

        optimization_result = minimize(
            loss,
            x0=np.array([1, 1]),
            # Our recommendation is to use:
            # - 'Nelder-Mead' if no gradient information is used
            # - 'BFGS' is gradient information is used
            method="BFGS",
            jac=loss.grad,
            args=(t, np.log(y)),
        )

        # Verify that learned parameters equal original parameters
        theta_estimated = np.array(optimization_result.x)
        assert np.allclose(theta_estimated, theta)


class TestFullCurveFittingExamplesWithARP:
    @pytest.mark.parametrize("production,days_on", PRODUCTION_AND_DAYS_ON)
    def test_transformed_curve_fit_invariance_constant_function(
        self, production, days_on
    ):
        """Test invariance between aggregated and daily data sets
        with respect to fitting a constant function."""

        def curve_func(x, params):
            """The mean function, i.e., the prediction is a constant."""
            return params

        production, days_on = np.array(production), np.array(days_on)

        # RMSE loss
        loss = CurveLoss(curve_func=curve_func, p=2.0)

        # Transform the data and fit to it. This data is assumed to be on a
        # monthly resolution, with 30 days per month. First we convert the
        # days_on per month to fraction of time on:
        time_on = days_on / 30
        cumulative_days_on, monthly_production, time_on = to_producing_time(
            time_on, production, center_intervals=True
        )
        # Then we convert to daily production rate:
        daily_production = monthly_production / 30

        # The third argument is the weight, since the loss signature is (x, y, weights)
        results = minimize(
            loss, x0=[0], args=(cumulative_days_on, daily_production, time_on)
        )
        params1 = results.x

        # Unravel the dataset to a daily level
        daily_production = np.repeat(production / days_on, days_on)
        cumulative_days_on = np.cumsum(np.array([1] * np.sum(days_on)))
        results = minimize(loss, x0=[0], args=(cumulative_days_on, daily_production))
        params2 = results.x

        assert np.allclose(params1, params2)

    @pytest.mark.parametrize("theta", THETAS)
    def test_that_curve_fitting_recovers_original_parametrization(self, theta):
        # Create a curve
        q_1, h, D = Arps(*theta).original_parametrization()
        t = np.linspace(0, 100, num=2**10)
        y = np.exp(Arps.eval_log_original_parametrization(t, q_1, h, D))

        # Learn parameters
        def log_Arps(t, theta1, theta2, theta3):
            return Arps(theta1, theta2, theta3).eval_log(t)

        loss = CurveLoss(curve_func=log_Arps, p=2.0)
        optimization_result = minimize(
            loss, x0=np.array([1, 1, 1]), method="BFGS", args=(t, np.log(y))
        )

        # Verify that learned parameters equal original parameters
        theta_estimated = np.array(optimization_result.x)
        assert np.allclose(
            [q_1, h, D], Arps(*theta_estimated).original_parametrization()
        )

    @pytest.mark.parametrize("theta", THETAS)
    def test_that_curve_fitting_with_gradient_works(self, theta):
        # Create a curve
        q_1, h, D = Arps(*theta).original_parametrization()
        t = np.linspace(0, 100, num=2**10)
        y = np.exp(Arps.eval_log_original_parametrization(t, q_1, h, D))

        # Learn parameters
        def log_ARPS(t, theta1, theta2, theta3):
            return Arps(theta1, theta2, theta3).eval_log(t)

        def log_ARPS_grad(t, theta1, theta2, theta3):
            return Arps(theta1, theta2, theta3).eval_grad_log(t)

        loss = CurveLoss(curve_func=log_ARPS, p=2.0, curve_func_grad=log_ARPS_grad)

        optimization_result = minimize(
            loss,
            x0=np.array([1, 1, 1]),
            # Our recommendation is to use:
            # - 'Nelder-Mead' if no gradient information is used
            # - 'BFGS' is gradient information is used
            method="BFGS",
            jac=loss.grad,
            args=(t, np.log(y)),
        )

        # Verify that learned parameters equal original parameters
        theta_estimated = np.array(optimization_result.x)
        assert np.allclose(
            [q_1, h, D], Arps(*theta_estimated).original_parametrization()
        )

    @pytest.mark.parametrize("theta", THETAS)
    def test_that_curve_fitting_with_gradient_outperforms_no_gradient_log_space(
        self, theta
    ):
        rng = np.random.default_rng(42)
        # Create a curve
        q_1, h, D = Arps(*theta).original_parametrization()
        t = np.linspace(0, 100, num=2**10)
        mu = Arps.eval_log_original_parametrization(t, q_1, h, D)
        y = np.exp(mu + rng.normal(size=len(mu)))

        # Learn parameters
        def log_ARPS(t, theta1, theta2, theta3):
            return Arps(theta1, theta2, theta3).eval_log(t)

        def log_ARPS_grad(t, theta1, theta2, theta3):
            return Arps(theta1, theta2, theta3).eval_grad_log(t)

        # No gradient information
        loss = CurveLoss(curve_func=log_ARPS, p=2.0)
        result_nograd = minimize(
            loss,
            x0=np.array([1, 1, 1]),
            method="Nelder-Mead",
            args=(t, np.log(y)),
        )

        # With gradient information
        loss = CurveLoss(curve_func=log_ARPS, p=2.0, curve_func_grad=log_ARPS_grad)
        result_grad = minimize(
            loss,
            x0=np.array([1, 1, 1]),
            method="BFGS",
            jac=loss.grad,
            args=(t, np.log(y)),
        )

        assert result_grad.fun < result_nograd.fun

    @pytest.mark.parametrize("theta", THETAS)
    def test_that_curve_fitting_with_gradient_outperforms_no_gradient_original_space(
        self, theta
    ):
        rng = np.random.default_rng(42)
        # Create a curve
        q_1, h, D = Arps(*theta).original_parametrization()
        t = np.linspace(0, 100, num=2**10)
        mu = Arps.eval_log_original_parametrization(t, q_1, h, D)
        y = np.exp(mu) + rng.normal(size=len(mu)) * 0.1

        # Learn parameters
        def ARPS(t, theta1, theta2, theta3):
            return np.exp(Arps(theta1, theta2, theta3).eval_log(t))

        def ARPS_grad(t, theta1, theta2, theta3):
            """Since (ln f)' = (1 / f) * f', we have f' = (ln f)' * f."""
            arps = Arps(theta1, theta2, theta3)
            return arps.eval_grad_log(t) * arps(t)

        # No gradient information
        loss = CurveLoss(curve_func=ARPS, p=2.0)
        result_nograd = minimize(
            loss,
            x0=np.array([1, 1, 1]),
            method="Nelder-Mead",
            args=(t, y),
        )

        # With gradient information
        loss = CurveLoss(curve_func=ARPS, p=2.0, curve_func_grad=ARPS_grad)
        result_grad = minimize(
            loss,
            x0=np.array([1, 1, 1]),
            method="BFGS",
            jac=loss.grad,
            args=(t, y),
        )

        # Close enough!
        assert result_grad.fun < result_nograd.fun + 1e-6


class TestNumerics:
    @pytest.mark.parametrize("x", np.logspace(-5, 2, num=10))
    def test_log1pexp_equals_naive_in_safe_range(self, x):
        t = 3
        assert np.isclose(log1pexp(t, x), np.log1p(t * np.exp(x)))

    @pytest.mark.parametrize("x", np.logspace(-5, 2, num=10))
    def test_log1pexp_inputs_vector(self, x):
        x = 3.13
        t = np.arange(100)

        ans_vector = log1pexp(t, x)
        ans_loop = np.array([log1pexp(t_i, x) for t_i in t])

        assert np.allclose(ans_vector, ans_loop)

    @pytest.mark.parametrize("t", np.logspace(-5, 5, num=10))
    def test_log1xp_vs_naive_out_of_safe_range(self, t):
        x = 1000

        # Should overflow
        with np.errstate(over="warn"):
            with pytest.warns(RuntimeWarning, match="overflow encountered"):
                np.log1p(t * np.exp(x))

        # Should not overflow
        log1pexp(t, x)


if __name__ == "__main__":
    import pytest

    pytest.main(
        args=[
            __file__,
            "--doctest-modules",
            "-v",
            "--capture=sys",
            "-k test_ar1model_on_SODIR_data",
            "-x",
        ]
    )
