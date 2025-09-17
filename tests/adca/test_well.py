"""
Tests for Well and WellGroup.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dca.adca.well import Well, WellGroup
from dca.decline_curve_analysis import Exponential


def relative_error(x, x_hat):
    return abs(x - x_hat) / abs(x)


class TestWell:
    def test_expected_vs_P50_params_vs_evaluations(self):
        """If we get paramters for the P50/expected curve, or we get
        predictions - the results should match."""

        well = Well.generate_random(
            curve_model="constant", time_on=np.ones(100), seed=42
        )

        well = well.fit(half_life=100, prior_strength=1e-3)

        # Curve parameters
        params = well.get_params(q=[0.5])

        # Forecast
        df_forecast = well.to_df(forecast_periods=10, q=[0.5])

        # Compare expected
        assert np.isclose(
            df_forecast.forecasted_production.dropna().mean(), params["C"]
        )

        # Compare P50
        assert np.isclose(
            df_forecast.forecasted_production_P50.dropna().mean(), params["C_P50"]
        )

    def test_that_half_life_matters(self):
        # Generate a random well and fit it
        time_on = np.ones(100)
        well = Well.generate_random(
            curve_model="arps", seed=2, preprocessing="producing_time", time_on=time_on
        )

        well.fit(p=2.0, half_life=None, prior_strength=0.0, split=1.0)
        thetas1 = np.array(well.curve_parameters_)

        well.fit(p=2.0, half_life=10, prior_strength=0.0, split=1.0)
        thetas2 = np.array(well.curve_parameters_)

        assert np.linalg.norm(thetas1 - thetas2) > 0.1

    @pytest.mark.parametrize("time_on", [0.1, 0.33, 0.5, 1])
    def test_simulation_vs_prediction_means(self, time_on):
        # Generate a random well and fit it
        time_on = np.ones(100)
        x = np.cumsum(time_on) - time_on / 2
        well = Well.generate_random(
            curve_model="arps", seed=2, preprocessing="producing_time", time_on=time_on
        )
        well.fit(p=2.0, half_life=None, prior_strength=1e-6, split=1.0)

        # Predict the mean pointwise
        y_mean = well.predict(x, time_on=time_on, q=None)

        # Predict the mean by simulations
        y_mean_sim = well.simulate(x, time_on=time_on, seed=3, simulations=9999)
        y_mean_sim_avg = np.mean(y_mean_sim, axis=0)

        # The result should be roughly the same
        assert np.allclose(y_mean, y_mean_sim_avg, rtol=0.01)

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("time_on", [0.33, 0.5, 1])
    @pytest.mark.parametrize("scale", [0.33, 1, 3])
    def test_std_scaling_constant_model(self, seed, time_on, scale):
        # Create well data: log(production/time_on) = N(0, scale)
        seed = seed + int(time_on * 10000) + int(scale * 1000)
        rng = np.random.default_rng(seed)
        n = 999
        time_on_arr = np.ones(n, dtype=float) * time_on
        noise = rng.normal(loc=0, scale=scale, size=n)
        production = time_on_arr * np.exp(noise)

        # Create a well, fit a constant model
        time = pd.period_range(start="2020-01-01", periods=n, freq="D")
        well = Well(
            production=production,
            time_on=time_on_arr,
            time=time,
            curve_model="constant",
            preprocessing="producing_time",
        )

        well.fit(p=2.0, half_life=None, prior_strength=0, split=1.0)

        # The mean has distribution E +/- (scale / sqrt(n))
        assert np.isclose(0, well.curve_parameters_[0], atol=scale / np.sqrt(n) * 3)

        # We attempt to infer the standard deviation on the "unit scale",
        # meaning that we scale by sqrt(time_on).
        assert np.isclose(scale**2 / time_on, well.std_**2, rtol=0.175)

    @pytest.mark.parametrize("split", [1.0, 0.5])
    @pytest.mark.parametrize("logscale", [True, False])
    @pytest.mark.parametrize("prediction", [True, False])
    @pytest.mark.parametrize("q", [None, [0.5]])
    def test_well_plot(self, split, logscale, prediction, q):
        """Smoketest - make sure every combination of arguments passes."""

        # Generate a random well and fit it
        well = Well.generate_random(
            n=100, curve_model="arps", seed=2, preprocessing="producing_time"
        )
        well.fit(p=2.0, half_life=None, prior_strength=1e-6, split=split)

        # Plot it
        well.plot(split=split, logscale=logscale, prediction=prediction, q=q)
        plt.close("all")

    @pytest.mark.parametrize("split", [1.0, 0.5])
    @pytest.mark.parametrize("ax", [True, None])
    @pytest.mark.parametrize("logscale", [True, False])
    @pytest.mark.parametrize("prediction", [True, False])
    @pytest.mark.parametrize("q", [None, [0.5]])
    def test_well_plot_cumulative(self, split, ax, logscale, prediction, q):
        """Smoketest - make sure every combination of arguments passes."""

        if ax is True:
            _, ax = plt.subplots()

        # Generate a random well and fit it
        well = Well.generate_random(
            n=100, curve_model="arps", seed=2, preprocessing="producing_time"
        )
        well.fit(p=2.0, half_life=None, prior_strength=1e-6, split=split)

        # Plot it
        well.plot_cumulative(
            split=split, ax=ax, logscale=logscale, prediction=prediction, q=q
        )
        plt.close("all")

    def test_splitting(self):
        time = pd.period_range(start="2020-01-01", periods=10, freq="D")
        production = 12 - np.arange(10)
        time_on = np.ones_like(time)
        well1 = Well(
            id=1,
            time=time,
            production=production,
            time_on=time_on,
            preprocessing="calendar_time",
        )
        well2 = Well(
            id=1,
            time=time,
            production=production,
            time_on=time_on,
            preprocessing="producing_time",
        )

        assert well1.get_split_index(0) == well2.get_split_index(0)
        assert well1.get_split_index(1 / 10) == well2.get_split_index(1 / 10)
        assert well1.get_split_index(5 / 10) == well2.get_split_index(5 / 10)
        period = pd.Period("2020-04-01")
        assert well1.get_split_index(period) == well2.get_split_index(period)

    def test_well_fit_calendar_time(self):
        production = np.array(
            [0.0, 5.764, 0.0, 1.875, 0.727, 0.339, 0.245, 0.108, 0.053, 0.026]
        )
        time_on = np.array(
            [0.0, 0.938, 0.0, 0.813, 0.95, 0.95, 0.855, 0.996, 0.936, 0.901]
        )
        time = pd.period_range(start="2020-01-01", periods=len(time_on), freq="D")
        well = Well.from_dirty_data(
            id=1,
            segment=1,
            production=production,
            time_on=time_on,
            time=time,
            preprocessing="calendar_time",
        )

        well = well.fit(p=2.0, half_life=10, prior_strength=1e-4)
        theta = np.array([2.537774, 0.3890166, -9.479001])
        assert np.allclose(well.curve_parameters_, theta, atol=0.01)

    def test_well_fit_producing_time(self):
        production = np.array(
            [0.0, 5.764, 0.0, 1.875, 0.727, 0.339, 0.245, 0.108, 0.053, 0.026]
        )
        time_on = np.array(
            [0.0, 0.938, 0.0, 0.813, 0.95, 0.95, 0.855, 0.996, 0.936, 0.901]
        )
        time = pd.period_range(start="2020-01-01", periods=len(time_on), freq="D")
        well = Well.from_dirty_data(
            id=1,
            segment=1,
            production=production,
            time_on=time_on,
            time=time,
            preprocessing="producing_time",
        )

        well = well.fit(p=2.0, half_life=10, prior_strength=1e-4)
        theta = np.array([2.2289823, -0.0763643, -1.9125074])
        assert np.allclose(well.curve_parameters_, theta, atol=0.01)


@pytest.mark.parametrize("mean", [0.1, 0.5, 1, 5])
@pytest.mark.parametrize("std", [0.5, 1, 2])
def test_DF_estimates_are_correct_assuming_perfect_inference(mean, std):
    """Sanity check - tests are values in DF are reasonable, given that
    the inference was correct (injecting true parameters into Well)."""

    well = Well.generate_random(
        time_on=np.ones(100),
        curve_model="constant",
        freq="D",
        curve_parameters=(mean,),
        std=std,
        seed=0,
    )
    well.fit(p=2.0, half_life=None, prior_strength=1e-8)

    # Inject true parameters into the well instance (assume perfect inference)
    well.std_ = std
    well.curve_parameters_ = (mean,)

    # Forecast 100 periods
    df = well.to_df(forecast_periods=100, q=[0.25, 0.5, 0.75])

    # Simulate directly from the simple constant model
    rng = np.random.default_rng(42)
    log_y = rng.normal(loc=std, scale=mean, size=(100, 999))
    simulations = np.sum(well.production) + np.cumsum(np.exp(log_y), axis=0)

    P25_obs = np.mean(df.cumulative_production_P25.values[100:] > simulations.T)
    assert np.isclose(P25_obs, 0.25, atol=1.0)

    P50_obs = np.mean(df.cumulative_production_P50.values[100:] > simulations.T)
    assert np.isclose(P50_obs, 0.50, atol=1.0)

    P75_obs = np.mean(df.cumulative_production_P75.values[100:] > simulations.T)
    assert np.isclose(P75_obs, 0.75, atol=1.0)


@pytest.mark.parametrize("n", [10, 25, 50, 100])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_that_to_df_is_consistent(n, seed):
    well = Well.generate_random(n=n, curve_model="exponential", freq="D", seed=seed)

    # Fit the well on all data
    well.fit(p=2.0, half_life=None, prior_strength=1e-3)

    df = well.to_df(forecast_periods=25, q=[0.25, 0.5, 0.75])

    def allclose_nan(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        return np.allclose(a[mask], b[mask])

    # Check that cumulative production adds up.
    prod_a = (df.production.fillna(0) + df.forecasted_production.fillna(0)).cumsum()
    prod_a = df.cumulative_production
    assert allclose_nan(prod_a, prod_a)

    # Check pointwise properties of forecasted values
    assert (
        df["forecasted_production_P25"].dropna()
        < df["forecasted_production_P50"].dropna()
    ).all()
    assert (
        df["forecasted_production_P50"].dropna()
        < df["forecasted_production_P75"].dropna()
    ).all()
    assert (
        df["forecasted_production_P25"].dropna() < df["forecasted_production"].dropna()
    ).all()
    assert (
        df["forecasted_production"].dropna() < df["forecasted_production_P75"].dropna()
    ).all()

    # Check pointwise properties of forecasted values
    assert (
        df["cumulative_production_P25"].dropna()
        <= df["cumulative_production_P50"].dropna()
    ).all()
    assert (
        df["cumulative_production_P50"].dropna()
        <= df["cumulative_production_P75"].dropna()
    ).all()
    assert (
        df["cumulative_production_P25"].dropna() <= df["cumulative_production"].dropna()
    ).all()
    assert (
        df["cumulative_production"].dropna() <= df["cumulative_production_P75"].dropna()
    ).all()

    # Smoketests - nothing should go wrong
    for forecast_periods, q in itertools.product([0, 1, 2, 100], [None, [0.5]]):
        df = well.to_df(forecast_periods=forecast_periods, q=q)
        assert len(df) == len(well) + forecast_periods


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("half_life", [None, 250, 125, 62.5])
@pytest.mark.parametrize("uptime", [0.9, 0.66, 0.33])
def test_standard_deviation_with_little_time_on(seed, half_life, uptime):
    time_on = np.ones(250)

    # This well is always on
    well1 = Well.generate_random(
        time_on=time_on, seed=seed, preprocessing="producing_time"
    )
    well1.fit(p=2.0, half_life=half_life, prior_strength=1e-4)
    std1 = well1.std_

    # This well is not always on
    well2 = Well.generate_random(
        time_on=time_on * uptime, seed=seed, preprocessing="producing_time"
    )
    well2.fit(p=2.0, half_life=half_life, prior_strength=1e-4)
    std2 = well2.std_

    # The uncertainty is expressed in time units of one period, so both
    # of these uncertainties should be very similar.
    assert np.isclose(std1, std2, rtol=0.05)


@pytest.mark.parametrize("seed", range(50))
def test_that_estimated_parameters_are_close(seed):
    rng = np.random.default_rng(seed + 99)

    # Parameters are tuned so they make sense on a grid of length 100
    curve_parameters = rng.normal(size=2, loc=[3, 4], scale=0.5)
    n = 100
    std = 0.05  # std here is tied to relative tolerances below

    well = Well.generate_random(
        n=n,
        curve_model="exponential",
        preprocessing="producing_time",
        curve_parameters=curve_parameters,
        seed=seed,
        std=std,
    )
    well.fit(p=2.0, half_life=None, prior_strength=1e-6)

    estimated = np.array(well.curve_parameters_)
    assert np.allclose(curve_parameters, estimated, rtol=0.05)
    assert np.isclose(std, well.std_, rtol=0.5)


@pytest.mark.parametrize("seed", range(10))
def test_that_EUR_is_the_sum_of_the_mean_curve(seed):
    rng = np.random.default_rng(seed)
    n = 75
    C, k = 10 + rng.normal(0, scale=1), 0.1 + rng.normal(0, scale=0.01)

    def model(t, C, k, std):
        f = Exponential.from_original_parametrization(C, k)
        return np.exp(f.eval_log(t) + rng.normal(scale=std, size=len(t)))

    # The true model is y = np.exp(t + epsilon)
    t = np.arange(0, n) + 0.5
    y = model(t, C, k, std=0.0)
    time = pd.period_range(start="2020-01-01", periods=n, freq="D")
    well = Well(id=1, time=time, production=y, curve_model="exponential")

    # Verify that we learn the parameters
    well.fit(p=2.0, phi=0, half_life=None, split=0.33, prior_strength=1e-12)

    C_hat, k_hat = well.curve_model(*well.curve_parameters_).original_parametrization()
    assert relative_error(C, C_hat) < 1e-5
    assert relative_error(k, k_hat) < 1e-5
    assert np.abs(well.std_) < 1e-5

    # Compute the expected EUR with std=1
    EUR = np.array([model(t, C, k, std=1) for _ in range(3333)])

    # Set true parameters
    well.std_ = 1.0
    well.curve_parameters_ = Exponential.from_original_parametrization(C, k).thetas

    # Verify pointwise convergence
    time_on = np.ones_like(t)
    MAE_mean = np.mean(
        np.abs(EUR.mean(axis=0) - well.predict(t, time_on=time_on, q=None))
    )
    assert MAE_mean < 0.1

    # Verify that MEAN curve is more appropriate than MEDIAN curve
    MAE_median = np.mean(
        np.abs(EUR.mean(axis=0) - well.predict(t, time_on=time_on, q=0.5))
    )
    assert MAE_mean < MAE_median

    # Verify the sums
    EUR_pred = np.sum(well.predict(t, time_on=time_on, q=None))
    assert relative_error(EUR.sum(axis=1).mean(), EUR_pred) < 0.02


def test_wellgroup_aggegration():
    p1 = np.array([100, 50, 100])
    p2 = np.array([100, 200])
    p3 = np.array([100])

    t1 = np.array([0.5, 0.3, 0.6])
    t2 = np.array([0.2, 0.4])
    t3 = np.array([0.1])

    w1 = Well(
        time=pd.period_range(start="2020-01-01", periods=3, freq="D"),
        production=p1,
        time_on=t1,
        id="A",
    )

    w2 = Well(
        time=pd.period_range(start="2020-01-01", periods=2, freq="D"),
        production=p2,
        time_on=t2,
        id="B",
    )

    w3 = Well(
        time=pd.period_range(start="2020-01-01", periods=1, freq="D"),
        production=p3,
        time_on=t3,
        id="C",
    )

    # Sum all three
    summed = WellGroup([w1, w2, w3]).aggregate(["A", "B", "C"])[0]

    # Production is easy -> just sum production for each day
    assert np.allclose(summed.production, [300, 250, 100])

    # Time on is a weighted harmonic average
    T1 = (300) / (100 / 0.5 + 100 / 0.2 + 100 / 0.1)
    T2 = (250) / (50 / 0.3 + 200 / 0.4)
    T3 = (100) / (100 / 0.6)

    assert np.allclose(summed.time_on, [T1, T2, T3])

    # Check trivial aggregation
    w2.segment = 2
    summed = WellGroup([w1, w2, w3]).aggregate(["A"])[-1]  # Ends up last
    assert np.allclose(summed.production, p1)
    assert np.allclose(summed.time_on, t1)


if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys", "-x"])
