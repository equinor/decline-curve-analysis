"""
Test timeseries preprocessing.
"""

import numpy as np
import pytest

from dca.timeseries import (
    TimeSeriesInterpolator,
    preprocess_timeseries,
    to_producing_time,
)


class TestTimeSeriesInterpolator:
    def test_missing_production_constant_rate(self):
        # The production rate is constant [10, 10, 10]
        production = np.array([10, np.nan, 3])  # Missing value: 5
        time_on = np.array([1, 0.5, 0.3])
        tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        time_on, production = tsi.interpolate()
        assert np.allclose(production, [10, 5, 3])

    def test_missing_production_exponential_rate(self):
        # The production rate is [100, 10, 1]
        production = np.array([100, np.nan, 1])
        time_on = np.array([1, 1, 1])

        # Logspace = True will infer 10 instead of 50.5
        tsi = TimeSeriesInterpolator(
            time_on=time_on, production=production, logspace=True
        )
        time_on, production = tsi.interpolate()
        assert np.allclose(production, [100, 10, 1])

        # Logspace = False will infer 50.5 instead of 10
        production = np.array([100, np.nan, 1])
        time_on = np.array([1, 1, 1])
        tsi = TimeSeriesInterpolator(
            time_on=time_on, production=production, logspace=False
        )
        time_on, production = tsi.interpolate()
        assert np.allclose(production, [100, 50.5, 1])

    def test_missing_time_on_constant_rate(self):
        # The production rate is constant [10, 10, 10]
        production = np.array([10, 6, 2])
        time_on = np.array([1, np.nan, 0.2])  # Missing value: 0.6
        tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        time_on, production = tsi.interpolate()
        assert np.allclose(time_on, [1.0, 0.6, 0.2])

    def test_missing_production_linear_rate(self):
        # The production rate is linear [30, np.nan, 10]
        production = np.array([30, np.nan, 2])
        time_on = np.array([1, 0.5, 0.2])

        # In the first period (x=0.5) the production rate is 30
        # In the third period (x=1.6) the production rate is 10
        # Interpolating at x=1.25 gives production rate of 16.3636...
        # Since time_on=0.5 in this time step, the production is 8.1818...
        tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        time_on, production = tsi.interpolate()
        assert np.allclose(production, np.array([30.0, 8.18181818, 2.0]))

    def test_that_time_on_is_capped_at_one(self):
        time_on = np.array([1, np.nan, 1])
        production = np.array([20, 30, 20])
        tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        time_on, production = tsi.interpolate()
        assert np.allclose(time_on, np.array([1.0, 1.0, 1.0]))

    def test_that_no_extrapolation_occurs(self):
        time_on = np.array([np.nan, 0.8, 0.6, 0.4])
        production = np.array([24, 16, 9.6, np.nan])
        tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        time_on, production = tsi.interpolate()
        assert np.allclose(time_on, np.array([np.nan, 0.8, 0.6, 0.4]), equal_nan=True)
        assert np.allclose(
            production, np.array([24.0, 16.0, 9.6, np.nan]), equal_nan=True
        )

    def test_missing_time_on_linear_rate(self):
        # The production rate is linear [30, np.nan, 10]
        production = np.array([30, 10, 2])
        time_on = np.array([1, np.nan, 0.2])

        tsi = TimeSeriesInterpolator(time_on=time_on, production=production)

        # In the first period (x=0.5) the production rate is 30
        # In the third period (x=1+???+0.2/2) the production rate is 10
        # But the x-value is unknown, because we do not know time_on!
        # Assuming that the unknown x-value is 1, we can interpolate between:
        # (x=0.5, y=30) and (x=2.1, y=10) at x=1.25 => rate = 17.5
        # Then time_on = production / rate = 10 / 17.5 = 0.5714285
        x = np.array([0.5, 1.5, 2.1])
        time_on_inp = tsi.interpolate_time_on(time_on, production, x=x)
        assert np.allclose(time_on_inp, np.array([1.0, 0.57142857, 0.2]))

        # This gives us a new estimate of time_on, which can in turn be used to
        # create an even better grid:
        x = np.array([0.5, 1 + 0.57142857 / 2, 1 + 0.57142857 + 0.1 / 2])

        # The estimate for time_on can now be updated
        time_on_inp = tsi.interpolate_time_on(time_on, production, x=x)
        assert np.allclose(time_on_inp, np.array([1.0, 0.62549801, 0.2]))

        # If we keep going, we end up converging to 0.6:
        time_on, production = tsi.interpolate()
        assert np.allclose(time_on, np.array([1.0, 0.6, 0.2]))

    @pytest.mark.parametrize("seed", range(10))
    def test_that_true_model_is_found(self, seed):
        rng = np.random.default_rng(42)
        intercept, slope = rng.normal(loc=[10, 0])

        # The production rate is given by f(x) = intercept - slope * x
        time_on = np.array([0.4, 0.6, 0.8, 1.0])
        x = np.cumsum(time_on) - time_on / 2
        production_rate = intercept - slope * x
        production = production_rate * time_on

        # Fill with some missing values:
        prod_NaN = production.copy()
        time_NaN = time_on.copy()
        prod_NaN[1] = np.nan
        time_NaN[2] = np.nan

        tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        time_on_est, production_est = tsi.interpolate()

        # The model should recover the true values when the underlying
        # production rate is linear over the range of missing values
        assert np.allclose(time_on_est, time_on)
        assert np.allclose(production_est, production)


class TestToProducingTime:
    """Test the function `to_producing_time`."""

    def test_basic_example(self):
        days_on = np.array([1, 0.5])
        production = np.array([100, 40])
        cumulative_days_on, daily_production, _ = to_producing_time(
            days_on, production, center_intervals=False
        )
        assert np.allclose(cumulative_days_on, np.array([1, 1.5]))
        assert np.allclose(daily_production, np.array([100, 80]))

    def test_basic_example_with_centering(self):
        days_on = np.array([1, 0.4])
        production = np.array([100, 40])
        cumulative_days_on, daily_production, _ = to_producing_time(
            days_on, production, center_intervals=True
        )
        assert np.allclose(cumulative_days_on, np.array([0.5, 1.2]))
        assert np.allclose(daily_production, np.array([100, 100]))


class TestTimeseriesPreprocessing:
    """Test the preprocessing function."""

    def test_that_NaNs_are_removed(self):
        time_on, production = np.array([1, np.nan]), np.array([100, np.nan])

        processed_days_on, processed_production, mask = preprocess_timeseries(
            time_on, production
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([1]))
        assert np.allclose(processed_production, np.array([100]))

    def test_that_NaNs_in_days_on_are_handled(self):
        time_on, production = np.array([1, np.nan]), np.array([100, 0])

        # NaN and 0
        processed_days_on, processed_production, mask = preprocess_timeseries(
            time_on, production
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([1]))
        assert np.allclose(processed_production, np.array([100]))

        # NaN and > 0
        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([1, np.nan]), np.array([100, 30])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([1, 1]))
        assert np.allclose(processed_production, np.array([100, 30]))

    def test_that_initial_NaNs_in_production_on_are_handled(self):
        # Initial NaN
        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([1, 0.5]), np.array([np.nan, 100])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([0.5]))
        assert np.allclose(processed_production, np.array([100]))

    def test_that_initial_zeros_in_production_are_handled(self):
        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([1, 0.5]), np.array([0, 100])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([0.5]))
        assert np.allclose(processed_production, np.array([100]))

    def test_that_zeros_in_days_on_are_handled(self):
        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([0, 0.5]), np.array([50, 100])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([1, 0.5]))
        assert np.allclose(processed_production, np.array([50, 100]))

        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([0, 0.5]), np.array([0, 100])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([0.5]))
        assert np.allclose(processed_production, np.array([100]))

        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([0, 0.5]), np.array([np.nan, 100])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([0.5]))
        assert np.allclose(processed_production, np.array([100]))

    def test_that_when_both_days_on_and_production_are_zero_they_are_removed(self):
        time_on = np.array([1, 0, 1])
        production = np.array([10, 0, 10])

        processed_days_on, processed_production, mask = preprocess_timeseries(
            time_on, production
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([1, 1]))
        assert np.allclose(processed_production, np.array([10, 10]))

    def test_that_zero_production_with_days_on_is_removed(self):
        # This was up for discussion, but we decided to remove observations of
        # zero production with positive days on, since we're predicting
        # PRODUCING days, and when there is no production it's not a producing month
        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([1, 1, 1]), np.array([10, 0, 10])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([1, 1]))
        assert np.allclose(processed_production, np.array([10, 10]))

    def test_that_NaN_production_with_days_on_is_removed(self):
        processed_days_on, processed_production, mask = preprocess_timeseries(
            np.array([1, np.nan, 1]), np.array([10, 0, 10])
        )
        processed_days_on, processed_production = (
            processed_days_on[mask],
            processed_production[mask],
        )
        assert np.allclose(processed_days_on, np.array([1, 1]))
        assert np.allclose(processed_production, np.array([10, 10]))


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
