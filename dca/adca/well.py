"""
A short tutorial on the Well class
==================================

A Well can be created by specifying its ID, time range, production, and time on.
A curve can be fit to the well data using the fit() method with hyperparameters
such as p, half_life, and prior_strength.

The curve parameters can be retrieved using the `curve_parameters_` attribute.
The well can also be scored on a test set and converted to a DataFrame.
The data can be cleaned using the `Well.clean_data()` method before creating a well.

The cleaned data can be passed to the get_curve_data() method to
obtain the data that will be used for curve fitting.


Creating a Well and fitting a curve
-----------------------------------

This is how we create a Well.

>>> well = Well(time=pd.period_range(start="2020-01-01", periods=6, freq="D"),
...           production=np.array([256, 128, 64, 32, 16, 8]),
...           time_on=np.array([1, 0.5, 1, 1, 0.5, 1]),
...           curve_model="exponential", preprocessing="calendar_time")

When a Well is created, no curve has yet been fit to it.

>>> well.is_fitted()
False

To fit a curve, call `fit()` with hyperparameters.
Wells are immutable, so a new instance is returned.

>>> well = well.fit(p=2.0, half_life=10, prior_strength=1e-2, split=1.0)

Curve parameters can now be retrieved:

>>> well.curve_parameters_
(6.2..., 0.3...)

A Well can be scored on a test set as follows (here using RMSE):

>>> split = 0.5 # Use a 50/50 split of the data
>>> well = well.fit(p=2.0, half_life=10, prior_strength=0, split=split)
>>> well.score(split=split, p=2, sigma=1, phi=0)
1.717...

A Well can be converted to a DataFrame:

>>> well.to_df().drop(columns=["segment"])
  well_id        time  production  time_on
0       1  2020-01-01       256.0      1.0
1       1  2020-01-02       128.0      0.5
2       1  2020-01-03        64.0      1.0
3       1  2020-01-04        32.0      1.0
4       1  2020-01-05        16.0      0.5
5       1  2020-01-06         8.0      1.0

We can also store the forecast in a DataFrame. We do not get perfect results
due to the weak prior. If we remove the prior we predict exactly 4, 2, 1.

>>> well.to_df(forecast_periods=3).drop(columns=["well_id", "segment", "time"])
   production  time_on  forecasted_production  cumulative_production
0       256.0      1.0                    NaN                  256.0
1       128.0      0.5                    NaN                  384.0
2        64.0      1.0                    NaN                  448.0
3        32.0      1.0                    NaN                  480.0
4        16.0      0.5                    NaN                  496.0
5         8.0      1.0                    NaN                  504.0
6         NaN      NaN                    4.0                  508.0
7         NaN      NaN                    2.0                  510.0
8         NaN      NaN                    1.0                  511.0

Cleaning a Well
---------------

Here are some questionable data:

>>> time = pd.period_range(start="2020-01-01", periods=6, freq="D")
>>> production = np.array([0,    60, np.nan,    125, 60, -30])
>>> time_on = np.array(   [1,   0.5,      1, np.nan,  1,   1])

Attempting to create a Well with this data will fail:

>>> well = Well(time=time, production=production, time_on=time_on)
Traceback (most recent call last):
...
ValueError: Well data not clean...

We must clean the data *before* creating a Well.
The times series data are fixed by calling `clean_well_data`, and
human-readable errors are returned. The fix applied depends on the error.

>>> from dca.adca.utils import clean_well_data
>>> cleaned, log = clean_well_data(production=production, time_on=time_on,
...                               time=time, return_log=True)
>>> for error in log:
...     print(error)
('production > 0 AND time_on > 0', 'unchanged', 2)
('production in (<=0, NaN) XOR time_on in (<=0, NaN)', 'interpolated', 2)
('production in (0, NaN) XOR time_on in (0, Nan)', 'removed', 2)

Now we can create a Well using the cleaned data:

>>> well = Well(id=1, preprocessing="calendar_time", **cleaned)
>>> well.to_df()
  well_id  segment        time  production  time_on
0       1        1  2020-01-02   60.000000      0.5
1       1        1  2020-01-03  103.636364      1.0
2       1        1  2020-01-04  125.000000      1.0
3       1        1  2020-01-05   60.000000      1.0

The data that will be passed to curve fitting is:

>>> x, y, w = well.get_curve_data()
>>> x
array([0.5, 1.5, 2.5, 3.5])
>>> y
array([ 60.        , 103.63636364, 125.        ,  60.        ])
>>> w
array([1., 1., 1., 1.])

Preprocessing options
---------------------

There are three preprocessing options, and they decide if we fit on calendar
time, calendar time with production adjusted, or producing time.

>>> well = Well(time=pd.period_range(start="2020-01-01", periods=4, freq="D"),
...           production=np.array([64, 16, 16, 2]), # 64, 32, 16, 8
...           time_on=np.array([1, 0.5, 1, 0.25]))

Calendar time:

>>> well.copy(preprocessing="calendar_time").get_curve_data()
(array([0.5, 1.5, 2.5, 3.5]), array([64., 16., 16.,  2.]), array([1., 1., 1., 1.]))

Producing time:

>>> well.copy(preprocessing="producing_time").get_curve_data()
(array([0.5  , 1.25 , 2.   , 2.625]), array([64., 32., 16.,  8.]), array([1.  , 0.5 , 1.  , 0.25]))


Uncertainty and simulation
--------------------------

Create a new Well instance:

>>> rng = np.random.default_rng(3)
>>> n = 500
>>> x = np.linspace(0, 5, num=n)
>>> y = np.exp(5 - x + rng.normal(size=n, scale=0.1))
>>> time = pd.period_range(start="2020-01-01", periods=n, freq="D")
>>> well = Well(time=time, production=y, curve_model="exponential")

Fit on the first half of the data:

>>> well = well.fit(p=2.0, half_life=None, prior_strength=1e-6, split=0.5)

To predict on the second half. The method `get_train_test` returns
a tuple with ((x_train, y_train, w_train), (x_test, y_test, w_test)).

>>> (_, _, _), (x_test, y_test, w_test) = well.get_train_test(split=0.5)

Predict the P50 curve and check for bias in log-space:

>>> predictions = well.predict(x_test, time_on=w_test, q=0.5)
>>> float(np.mean(np.log(y_test) - np.log(predictions)))
0.0069...

Predict mean curve and check for bias in original space:

>>> y_pred = well.predict(x_test, time_on=w_test, q=None)
>>> float(np.mean(y_test - y_pred))
0.0421...

Check RMSE (data generated with scale=0.1, so this is quite accurate):

>>> predictions = well.predict(x_test, time_on=w_test, q=0.5)
>>> float(np.sqrt(np.mean(np.square(np.log(y_test) - np.log(predictions)))))
0.0986...

Estimated standard devaiation (the true standard deviation is scale=0.1):

>>> well.std_
0.10...

To predict P10 and P90 on the test set, we can do the following:

>>> import scipy as sp
>>> P10 = np.log(well.predict(x_test, time_on=w_test, q=0.1))
>>> P90 = np.log(well.predict(x_test, time_on=w_test, q=0.9))
>>> # TODO: these values are wrong
>>> float(np.mean(np.log(y_test) < P10))
0.084
>>> float(np.mean(np.log(y_test) >= P90))
0.1

To simulate EUR on the test set, we can use the `simulate` method:

>>> EURS = well.simulate_cumprod(x_test, seed=0, simulations=999)[:, -1]
>>> expected_EUR, std_EUR = float(np.mean(EURS)), float(np.std(EURS))
>>> expected_EUR, std_EUR
(1125.5..., 8.87...)
>>> actual_EUR = float(np.sum(y_test))
>>> actual_EUR
1135.96...

Let us do a more realistic example. We fit to all data, then forecast into the
future for twice as long.

>>> well = well.fit(p=2.0, half_life=None, prior_strength=1e-6, split=1.0)

>>> x_grid = well.forecasting_grid(periods=n)
>>> sim_EURs = well.simulate_cumprod(x_grid, seed=0, simulations=999)[:, -1]
>>> EURS = np.sum(y) + sim_EURs
>>> expected_EUR, std_EUR = float(np.mean(EURS)), float(np.std(EURS))
>>> expected_EUR, std_EUR
(14971.6..., 0.72...)
>>> float(np.sum(y))
14870.9...
"""

import collections
import copy
import functools
import logging
import numbers
from collections import UserList
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp

from dca.adca.utils import (
    PeriodIndexer,
    clean_well_data,
    pairwise,
    to_filename,
    transform_bounds,
    transform_parameters,
)
from dca.decline_curve_analysis import Arps, Constant, Exponential
from dca.models import AR1Model

COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
WELLGROUP_SEP = ","  # Character used to concatenate IDs in summed wells

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def fnum(x, precision=4):
    return np.format_float_positional(x, unique=False, precision=precision)


@dataclass
class Well:
    """A Well represents a time series for an actual well, or for a segment
    in a well. For instance, one physical well could be represented as 3
    different Well objects if there are 3 different segments."""

    model_map = {"arps": Arps, "exponential": Exponential, "constant": Constant}
    preprocessing_options = ("calendar_time", "producing_time")

    def __init__(
        self,
        *,
        production: npt.NDArray[float],
        time: npt.NDArray[object],
        time_on: npt.NDArray[float] = None,
        id: str = "1",
        segment: int = 1,
        preprocessing: str = "producing_time",
        curve_model: str = "arps",
    ):
        if time_on is None:
            time_on = np.ones_like(production)

        assert isinstance(production, np.ndarray)
        assert np.issubdtype(production.dtype, np.number)

        assert isinstance(time_on, np.ndarray)
        assert np.nanmin(time_on) >= 0
        assert np.nanmax(time_on) <= 1
        assert isinstance(time.dtype, pd.PeriodDtype)
        assert (time.sort_values().values == time.values).all(), "`time` must be sorted"
        assert np.all(np.diff([period.ordinal for period in time]) > 0), (
            "`time` must be unique"
        )
        assert isinstance(segment, numbers.Integral)
        assert len(production) == len(time_on) == len(time)
        assert isinstance(id, (str, numbers.Integral))

        self.id = str(id)
        self.production = np.array(production, dtype=float)
        self.time_on = np.array(time_on, dtype=float)
        self.time = pd.PeriodIndex(time.copy())
        self.segment = segment
        assert preprocessing in self.preprocessing_options
        self.preprocessing = preprocessing
        self.curve_model_str_ = str(curve_model)
        self.curve_model = self.model_map[curve_model]

        # Seed used in simulations inferred from the ID
        # We cannot use hash() since it is not deterministic between
        # runs of the Python interpreter. Therefore we use the bytes.
        bytes_ = str(self.id).encode("utf-8")
        self.seed_ = int.from_bytes(bytes_, byteorder="big") % 2**10

        # Verify that quality is good
        if not (np.all(self.production > 0)):
            raise ValueError("Well data not clean (`production` is <=0 or NaN)")

        # Verify that quality is good
        if not (np.all(self.time_on > 0)):
            raise ValueError("Well data not clean (`time_on` is <=0 or NaN)")

    @classmethod
    def from_dirty_data(
        cls,
        *,
        production: npt.NDArray[float],
        time: npt.NDArray[object],
        time_on: npt.NDArray[float] = None,
        id: str = "1",
        segment: int = 1,
        preprocessing: str = "producing_time",
        curve_model: str = "arps",
    ):
        cleaned = clean_well_data(production=production, time=time, time_on=time_on)
        return cls(
            production=cleaned.production,
            time=cleaned.time,
            time_on=cleaned.time_on,
            id=id,
            segment=segment,
            preprocessing=preprocessing,
            curve_model=curve_model,
        )

    def is_fitted(self) -> bool:
        """Check if a DCA model has been fitted to data."""
        return hasattr(self, "curve_parameters_")

    def get_split_index(self, split):
        """Get the index to split on.

        Examples
        --------

        Generate some data to split.

        >>> time = pd.period_range(start="2020-01-01", periods=10, freq="D")
        >>> production = 12 - np.arange(10)
        >>> time_on = np.ones(len(time))
        >>> well = Well(id=1, time=time, production=production, time_on=time_on)

        Ways to split it:

        >>> well.get_split_index(2)
        2
        >>> well.get_split_index(0.5)
        5
        >>> well.get_split_index(pd.Period("2020-01-01"))
        0
        >>> well.get_split_index(pd.Period("2020-01-06"))
        5

        Tests on cumulative time:

        >>> time = pd.period_range(start="2020-01-01", periods=4, freq="D")
        >>> production = np.array([10, 9, 8, 7])
        >>> time_on = np.array([1, 0.5, 0.25, 0.25])
        >>> well = Well(id=1, time=time, production=production, time_on=time_on)
        >>> well.get_split_index(0.0)
        0
        >>> well.get_split_index(0.5 - 1e-12), well.get_split_index(0.5 + 1e-12)
        (0, 1)
        >>> well.get_split_index(0.5)
        1
        >>> well.get_split_index(1.0)
        4

        """
        if isinstance(split, int):
            return split
        elif isinstance(split, float) and self.preprocessing == "calendar_time":
            assert 0.0 <= split <= 1.0
            return int(len(self) * split)
        elif isinstance(split, float) and self.preprocessing == "producing_time":
            assert 0.0 <= split <= 1.0
            cumtime = np.cumsum(self.time_on)
            # This mimics Python behavior for list slicing
            return int(np.searchsorted(cumtime, v=cumtime[-1] * split, side="right"))
        elif isinstance(split, pd.Period):
            return int(np.sum(self.time < split))
        else:
            raise TypeError("Arg 'split' must be int, float or pd.Period")

    def fit(
        self,
        *,
        # hyperparameters
        half_life: float,
        prior_strength: float,
        # parameters
        p=None,  # p-norm, or beta in generalized normal distribution (GND)
        sigma=None,  # scale parameter for GND (larger than 0)
        phi=None,  # autocorrelation parameter (between 0 and 1)
        # Default is to fit on all data
        split=1.0,
        neg_logpdf_prior=None,
    ) -> tuple:
        """Fit a DCA curve to the Well, return self.

        Examples
        --------
        >>> well = Well.generate_random(seed=0, preprocessing="producing_time",
        ...                             curve_model="exponential")
        >>> well = well.fit(p=2.0, half_life=20, prior_strength=1e-4)
        >>> well.curve_parameters_
        (3.6..., 1.3...)
        """
        # At the boundary the prior can evaluate to -infty, so clip the values
        epsilon = 1e-10
        p = p if p is None else min(max(1 + epsilon, p), 2 - epsilon)
        sigma = sigma if sigma is None else max(0 + epsilon, sigma)
        phi = phi if phi is None else min(max(0 + epsilon, phi), 1 - epsilon)

        half_life = np.inf if half_life is None else half_life

        # Split the data
        (x_train, y_train, w_train), _ = self.get_train_test(split)
        assert np.all(y_train > 0) and not np.any(np.isclose(y_train, 0.0))

        if np.isclose(np.sum(w_train), 0):
            raise ValueError(f"No training data for well: {self}")

        # Create a model
        model = AR1Model(curve_cls=self.curve_model)
        model.update(theta=None, sigma=sigma, phi=phi, p=p)

        opt_params = model.optimize(
            tau=w_train,
            t=x_train,
            y=y_train,
            prior_strength=prior_strength,
            half_life=half_life,
            neg_logpdf_prior=neg_logpdf_prior,
        )

        self.curve_parameters_ = tuple(float(t_j) for t_j in opt_params["theta"])
        params = {"p": p, "sigma": sigma, "phi": phi}
        self.parameters_ = params | opt_params  # Update inputs with learned values

        # Compute the pointwise standard deviation of the long-running AR1 process
        self.std_ = model.pointwise_std(
            p=self.parameters_["p"],
            sigma=self.parameters_["sigma"],
            phi=self.parameters_["phi"],
        )
        log.debug(f"Fitted {self}. Optimal parameters: {opt_params}")

        # Compute average time on
        hl_weights = model.get_weights(t=x_train, tau=w_train, half_life=half_life)
        self.avg_time_on_ = np.average(w_train, weights=hl_weights)

        return self

    def plot(self, split, *, ax=None, logscale=False, prediction=True, q=None):
        """Plot a well."""
        assert q is None or all(0 < q_i < 1 for q_i in q)
        q = [] if q is None else q

        # Create a figure and axis if and axis-instance was not given
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        well_name = to_filename(self.id.split(","))
        ax.set_title(f"Well '{well_name}' Segment '{self.segment}' Split: {split!r}")

        # Get train and test data to plot
        train, test = self.get_train_test(split)
        x_train, y_train, w_train = train
        x_test, y_test, w_test = test

        y_label = "Production / time"

        if logscale:
            y_train, y_test = np.log(y_train), np.log(y_test)
            y_label = f"log({y_label})"

        # Plot the DCA curve
        x_labels_all = self.time.values
        if prediction and self.is_fitted():
            # Set up the future grid
            freq = self.time.dtype.freq.freqstr
            periods = 36 if freq == "ME" else 360 * 3
            periods = max(periods, len(self.time) // 3)
            x_future, periods_future = self.forecasting_grid(
                periods, return_periods=True
            )

            # Concatenate test set with future
            x_labels_all = np.concatenate((x_labels_all, periods_future))

            # We will now create an equidistant grid from the start of observed
            # production (start of training data set) till the end of the future
            # prediction interval. We avoid using the possibly non-equidistant
            # values in `x`, arising from different values of `time_on`, because
            # non-equidistant values can be confusing for the user.
            # For time_on=0.1, there is very little uncertainty, so time_on=[1, 0.1, 1]
            # would lead to uncertainty envelopes that shrink, then go up again.
            # This is correct if the goal is to show pointwise uncertainty, but
            # is confusing, so instead we use the average time on.
            # This is only a visual problem. When ADCA predicts the future production,
            # we always use an equidistant grid, so then this is never a problem.

            # Create a grid from smallest to largest-ish value to evaluate on
            x_first, x_last = x_train[0], x_future[-1]
            step = self.avg_time_on_
            x_smooth = np.arange(start=x_first, stop=x_last + step, step=step)
            w_smooth = np.ones_like(x_smooth, dtype=float) * step

            # The expected curve
            y_smooth = self.predict(x_smooth, time_on=w_smooth, q=None)
            y_smooth = np.log(y_smooth) if logscale else y_smooth
            y_max = np.max(y_smooth)
            y_min = np.min(y_smooth)
            ax.plot(
                x_smooth,
                y_smooth,
                zorder=10,
                label=str(self.curve_model.__name__),
                color="black",
            )

            # Plot quantile curves
            for q_i in q:
                y_smooth = self.predict(x_smooth, time_on=w_smooth, q=q_i)
                y_smooth = np.log(y_smooth) if logscale else y_smooth
                y_max = max(y_max, np.max(y_smooth))
                y_min = min(y_min, np.min(y_smooth))
                ax.plot(
                    x_smooth,
                    y_smooth,
                    zorder=10,
                    color="black",
                    linestyle="--",
                )

        # Plot the individual data points
        x, _, _ = self.get_curve_data()

        # Compute size of dots. n=1000 -> s_base=3, n=25 -> s_base=25
        s_base = min(max(3, -0.02 * len(x) + 25), 25)
        ax.scatter(
            x_train,
            y_train,
            zorder=5,
            s=s_base * w_train,
            color=COLORS[0],
            label="Train",
        )
        ax.scatter(
            x_test, y_test, zorder=8, s=s_base * w_test, color=COLORS[1], label="Test"
        )

        # Attempt to set up a sensible y-axis
        _, y, w = self.get_curve_data()
        y = np.log(y[w > 0]) if logscale else y[w > 0]
        y_max = max(y_max, np.max(y)) if "y_max" in locals() else np.max(y)
        y_min = min(y_min, np.min(y)) if "y_min" in locals() else np.min(y)
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range / 10, y_max + y_range / 10)
        ax.set_ylabel(y_label)

        # Plot dates along x axis
        if self.preprocessing != "producing_time":
            # This is not very good, but it kind of works
            x_ticks = ax.get_xticks()[1:-1]
            ax.set_xticks(x_ticks)

            def clip(x):
                return min(max(0, int(x)), len(x_labels_all) - 1)

            ax.set_xticklabels(
                [x_labels_all[clip(x_t)] for x_t in x_ticks],
                rotation=45,
            )
            ax.set_xlabel(f"Time ({self.time.dtype.freq.freqstr})")
        else:
            # Plot values along x axis
            first_period = self.time[self.time_on > 0].min()
            ax.set_xlabel(f"Periods since first period with production: {first_period}")

        ax.grid(True, ls="--", zorder=0, alpha=0.33)
        ax.legend()
        fig.tight_layout()

        return fig, ax

    def plot_simulations(
        self, split, *, ax=None, logscale=False, q=None, simulations=1
    ):
        """Plot simulations from a well. Show `simulations` and compute quantiles
        using 999 simulations."""
        assert q is None or all(0 < q_i < 1 for q_i in q)
        q = [] if q is None else q
        assert self.is_fitted()
        assert simulations > 0

        # Create a figure and axis if and axis-instance was not given
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        well_name = to_filename(self.id.split(","))
        ax.set_title(
            f"Well '{well_name}' Segment '{self.segment}' Split: {repr(split)}"
        )

        # Get train and test data to plot
        train, test = self.get_train_test(split)
        x_train, y_train, w_train = train
        x_test, y_test, w_test = test

        y_label = "Production / time"

        if logscale:
            y_train, y_test = np.log(y_train), np.log(y_test)
            y_label = f"log({y_label})"

        # Plot the DCA curve
        x_labels_all = self.time.values

        # Set up the future grid
        freq = self.time.dtype.freq.freqstr
        periods = 36 if freq == "ME" else 360 * 3
        periods = max(periods, len(self.time) // 3)
        x_future, periods_future = self.forecasting_grid(periods, return_periods=True)

        # Concatenate test set with future
        x_labels_all = np.concatenate((x_labels_all, periods_future))

        # Create a grid from smallest to largest-ish value to evaluate on
        x_first, x_last = x_train[0], x_future[-1]
        step = self.avg_time_on_
        x_smooth = np.arange(start=x_first, stop=x_last + step, step=step)
        w_smooth = np.ones_like(x_smooth, dtype=float) * step

        # The expected curve
        y_smooth = self.predict(x_smooth, time_on=w_smooth, q=None)
        y_smooth = np.log(y_smooth) if logscale else y_smooth
        ax.plot(
            x_smooth,
            y_smooth,
            zorder=10,
            label=str(self.curve_model.__name__),
            color="black",
        )

        # Plot simulations
        x_sim_start = x_test[0] if len(x_test) > 0 else x_train[-1]
        x_sim_end = x_future[-1]
        x_simgrid = np.arange(start=x_sim_start, stop=x_sim_end + step, step=step)
        time_on_sim = np.ones_like(x_simgrid, dtype=float) * step

        # Compute the prev_eta value
        (x_train_eta, y_train_eta, _), _ = self.get_train_test(split)
        curve = self.curve_model(*self.curve_parameters_)
        prev_eta = np.log(y_train_eta[-1]) - curve.eval_log(x_train_eta[-1])

        # Simulate production rates
        y_sims = self.simulate(
            x=x_simgrid,
            time_on=time_on_sim,
            prev_eta=prev_eta,
            seed=self.seed_,
            simulations=999,
        )

        # Plot quantiles
        if q is not None:
            for y_sim in np.percentile(y_sims, q=np.array(q) * 100, axis=0):
                y_sim = np.log(y_sim) if logscale else y_sim
                ax.plot(
                    x_simgrid,
                    y_sim,
                    zorder=10,
                    color="black",
                    linestyle="--",
                )

        for i, y_sim in enumerate(y_sims[:simulations]):
            y_sim = np.log(y_sim) if logscale else y_sim
            label = "Simulations" if i == 0 else None
            ax.plot(
                x_simgrid,
                y_sim,
                alpha=1 / (0.5 + simulations),
                color=COLORS[2],
                label=label,
            )

        # Plot the individual data points
        x, _, _ = self.get_curve_data()

        ax.plot(
            x_train,
            y_train,
            zorder=5,
            color=COLORS[0],
            label="Train",
        )
        ax.plot(x_test, y_test, zorder=8, color=COLORS[1], label="Test")

        # Attempt to set up a sensible y-axis
        _, y, w = self.get_curve_data()
        y = np.log(y[w > 0]) if logscale else y[w > 0]
        y_range = np.max(y) - np.min(y)
        ax.set_ylim([np.min(y) - y_range / 10, np.max(y) + y_range / 10])
        ax.set_ylabel(y_label)

        # Plot dates along x axis
        if self.preprocessing != "producing_time":
            # This is not very good, but it kind of works
            x_ticks = ax.get_xticks()[1:-1]
            ax.set_xticks(x_ticks)

            def clip(x):
                return min(max(0, int(x)), len(x_labels_all) - 1)

            ax.set_xticklabels(
                [x_labels_all[clip(x_t)] for x_t in x_ticks],
                rotation=45,
            )
            ax.set_xlabel(f"Time ({self.time.dtype.freq.freqstr})")
        else:
            # Plot values along x axis
            first_period = self.time[self.time_on > 0].min()
            ax.set_xlabel(f"Periods since first period with production: {first_period}")

        ax.grid(True, ls="--", zorder=0, alpha=0.33)
        ax.legend()
        fig.tight_layout()

        return fig, ax

    def plot_cumulative(
        self, split, *, ax=None, logscale=False, prediction=True, q=None
    ):
        """Plot a well on cumulative scale."""
        assert q is None or all(0 < q_i < 1 for q_i in q)

        # Create a figure and axis if and axis-instance was not given
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        well_name = to_filename(self.id.split(","))
        ax.set_title(f"Well '{well_name}' Segment '{self.segment}' Split: {split!r}")

        # Get train and test data to plot
        train, test = self.get_train_test(split)
        x_train, y_train, w_train = train
        x_test, y_test, w_test = test

        # Go from production rates to production
        production_train = np.cumsum(y_train * w_train)
        production_test = production_train[-1] + np.cumsum(y_test * w_test)

        y_label = "Production"
        if logscale:
            y_label = f"log({y_label})"

        # Plot the DCA curve
        x_labels_all = self.time.values
        y_lim_max = 0
        if prediction and self.is_fitted():
            # Set up the future grid
            freq = self.time.dtype.freq.freqstr
            periods = 36 if freq == "ME" else 360 * 3
            periods = max(periods, len(self.time) // 3)
            x_future, periods_future = self.forecasting_grid(
                periods, return_periods=True
            )
            assert np.allclose(np.diff(x_future), 1.0)

            # Concatenate test set with future
            x_labels_all = np.concatenate((x_labels_all, periods_future))

            # Create a grid from smallest to largest value to evaluate on.
            # Cumulatives are only shown on training data and future data.
            x_first = x_future[0] if len(x_test) == 0 else x_test[0]
            x_last = x_future[-1]
            step = self.avg_time_on_
            x_smooth = np.arange(start=x_first, stop=x_last + step, step=step)

            # Compute prev-eta based on the last training data point
            curve = self.curve_model(*self.curve_parameters_)
            prev_eta = np.log(y_train[-1]) - curve.eval_log(x_train[-1])

            # Create simulations
            sims = self.simulate_cumprod(
                x_smooth, prev_eta=prev_eta, seed=self.seed_, simulations=999
            )

            # The expected curve
            y_smooth = production_train[-1] + np.average(sims, axis=0)

            # Update scale and take logs if desired
            y_smooth = np.log(y_smooth) if logscale else y_smooth
            y_lim_max = np.max(y_smooth)

            ax.plot(
                x_smooth,
                y_smooth,
                zorder=10,
                label=str(self.curve_model.__name__),
                color="black",
            )

            # Plot quantile curves
            if q is not None:
                y_sims = production_train[-1] + sims

                for y_sim in np.percentile(y_sims, q=np.array(q) * 100, axis=0):
                    y_sim = np.log(y_sim) if logscale else y_sim
                    y_lim_max = max(y_lim_max, np.max(y_sim))

                    ax.plot(
                        x_smooth,
                        y_sim,
                        zorder=10,
                        color="black",
                        linestyle="--",
                    )

        # Plot the individual data points
        x, _, _ = self.get_curve_data()

        # Compute size of dots. n=1000 -> s_base=3, n=25 -> s_base=25
        s_base = min(max(3, -0.02 * len(x) + 25), 25)

        if logscale:
            production_train = np.log(production_train)
            production_test = np.log(production_test)

        ax.scatter(
            x_train,
            production_train,
            zorder=5,
            s=s_base * w_train,
            color=COLORS[0],
            label="Train",
        )
        ax.scatter(
            x_test,
            production_test,
            zorder=8,
            s=s_base * w_test,
            color=COLORS[1],
            label="Test",
        )

        # Attempt to set up a sensible y-axis
        _, y, w = self.get_curve_data()
        y = np.cumsum(y)
        y = np.log(y[w > 0]) if logscale else y[w > 0]
        y_range = max(y_lim_max, np.max(y)) - np.min(y)
        ax.set_ylim([np.min(y) - y_range / 10, y_lim_max + y_range / 10])
        ax.set_ylabel(y_label)

        # Plot dates along x axis
        if self.preprocessing != "producing_time":
            # This is not very good, but it kind of works
            x_ticks = ax.get_xticks()[1:-1]
            ax.set_xticks(x_ticks)

            def clip(x):
                return min(max(0, int(x)), len(x_labels_all) - 1)

            ax.set_xticklabels(
                [x_labels_all[clip(x_t)] for x_t in x_ticks],
                rotation=45,
            )
            ax.set_xlabel(f"Time ({self.time.dtype.freq.freqstr})")
        else:
            # Plot values along x axis
            first_period = self.time[self.time_on > 0].min()
            ax.set_xlabel(f"Periods since first period with production: {first_period}")

        ax.grid(True, ls="--", zorder=0, alpha=0.33)
        ax.legend()
        fig.tight_layout()

        return fig, ax

    def relative_cumulative_error(self, split, q=None):
        """Yield tuples with (sum_forecast, sum_actual) for every q_in in q,
        evaluated over the test set. `time_on` from the test set is used."""
        q = [] if q is None else q
        assert isinstance(q, list), "q must be a list of quantiles in (0, 1)"
        assert all((q_i is None) or (0 < q_i < 1) for q_i in q)

        # Split the data
        (x, y, _), (x_test, y_test, w_test) = self.get_train_test(split=split)

        # Compute prev-eta based on the last data point observed
        curve = self.curve_model(*self.curve_parameters_)
        prev_eta = np.log(y[-1]) - curve.eval_log(x[-1])

        # No test data, return a perfect score
        if np.isclose(np.sum(w_test), 0):
            for q_i in q:
                yield 1.0, 1.0
            return

        # Simulate production rates
        sims = self.simulate(
            x=x_test,
            time_on=w_test,
            seed=self.seed_,
            prev_eta=prev_eta,
            simulations=999,
        )

        # The production is the production rate times the time elapsed.
        forecasted_values = np.sum(w_test * sims, axis=1)

        idx = self.get_split_index(split=split)
        train_prod, test_prod = self._split_and_remove_zero_time_on(
            self.production, idx=idx
        )
        actual_value = np.sum(test_prod)

        for q_i in q:
            # Compute the mean, or a percentile, over the distribution of values
            if q_i is None:
                forecasted_value = np.mean(forecasted_values)
            else:
                forecasted_value = np.percentile(forecasted_values, q=q_i)

            yield float(forecasted_value), float(actual_value)

    def to_df(self, forecast_periods: int = 0, q=None, simulations=999) -> pd.DataFrame:
        """Convert the Well to a DataFrame, optionally including a forecast.


        Parameters
        ----------
        forecast_periods : int, optional
            Number of periods to forecast. The default is 0.
        q : list, optional
            Quantiles between 0 and 1. The default is None.
        simulations : int, optional
            Number of simulations in forecsating. The default is 999.

        Returns
        -------
        pd.DataFrame
            A dataframe with historical data and a forecast.

        Examples
        --------
        >>> well = Well.generate_random(seed=42, preprocessing="producing_time")
        >>> well.to_df().head()
          well_id  segment        time  production   time_on
        0       1        1  2010-01-01    7.355008  0.974701
        1       1        1  2010-01-02    4.872264  0.920947
        2       1        1  2010-01-03    6.324433  0.984870
        3       1        1  2010-01-04    5.562379  0.964598
        4       1        1  2010-01-06    2.922971  0.997535
        """
        if isinstance(forecast_periods, pd.Period):
            forecast_periods = max(0, (forecast_periods - self.time.max()).n - 1)

        assert isinstance(forecast_periods, int) and (forecast_periods >= 0)
        q = [] if q is None else q  # Empty iterable
        assert isinstance(q, list), "q must be a list of quantiles in (0, 1)"
        assert all(0 < q_i < 1 for q_i in q)
        assert isinstance(simulations, int) and simulations >= 99

        # No forecast, only historical data
        if forecast_periods <= 0:
            return pd.DataFrame(
                {
                    "well_id": [self.id] * len(self),
                    "segment": [self.segment] * len(self),
                    "time": self.time,
                    "production": self.production,
                    "time_on": self.time_on,
                }
            )

        # Assume that we want to forecast
        assert self.is_fitted(), "Well must be fit to forecast"
        df_history = self.to_df(forecast_periods=0)
        x_grid, period_grid = self.forecasting_grid(
            periods=forecast_periods, return_periods=True
        )

        # PRF: Production rate forecasts
        time_on = 1.0 * np.ones_like(x_grid)
        prf = {
            "forecasted_production": self.predict(x_grid, time_on=time_on, q=None)
        }  # Expected
        for q_i in q:
            percentile = str(round(q_i * 100)).rjust(2, "0")
            prf[f"forecasted_production_P{percentile}"] = self.predict(
                x_grid, time_on=time_on, q=q_i
            )

        # CF: Cumulative forecasts
        def concat(y):
            """Concatenate future cumulative data x with historical data."""
            y_cumsum = np.cumsum(df_history.production.to_numpy())
            return np.concatenate((y_cumsum, y_cumsum[-1] + y))

        # Expected value and percentiles. To compute this we need to simulate,
        # since the percentiles of a weighted sum of log GND's has no simple equation.
        # Simulate future values, with noise given by N(0, std). Shape (sims, len(x))

        # Compute prev-eta based on the last data point observed
        curve = self.curve_model(*self.curve_parameters_)
        (x, y, _) = self.get_curve_data()
        prev_eta = np.log(y[-1]) - curve.eval_log(x[-1])

        # Simulate future cumulative production
        simulations = self.simulate_cumprod(
            x_grid, prev_eta=prev_eta, seed=self.seed_, simulations=simulations
        )

        # Average overall simulations
        mean_sim = np.average(simulations, axis=0)
        cf = {"cumulative_production": concat(mean_sim)}

        # Compute pointwise percentiles over the simulation period
        for q_i in q:
            percentile = str(round(q_i * 100)).rjust(2, "0")
            perc_sim = np.percentile(simulations, q=q_i * 100, axis=0)
            cf[f"cumulative_production_P{percentile}"] = concat(perc_sim)

        # Create dataframe with future predictions
        df_future = pd.DataFrame(
            {
                "time": period_grid,
                "well_id": [self.id] * forecast_periods,
                "segment": [self.segment] * forecast_periods,
            }
            | prf
        )

        # Concatenate historical and future data, assign cumulative forecasts
        return pd.concat([df_history, df_future], axis=0, ignore_index=True).assign(
            **cf
        )

    def __str__(self):
        return type(self).__name__ + f"(id='{self.id}', segment={self.segment})"

    def __repr__(self):
        return str(self)

    def get_curve_data(self):
        """Transform data and return (x, y, w) for curve fitting.

        Examples
        --------

        Generate some data:

        >>> time = pd.period_range(start="2020-01-01", periods=6, freq="D")
        >>> production = np.array([32, 8, 8, 1, 2, 1])
        >>> time_on = np.array([1, 0.5, 1, 0.25, 1, 1])
        >>> well = Well(id=1, time=time, production=production,
        ...             time_on=time_on, preprocessing="calendar_time")

        Preprocess on calendar time with no adjustments for production rate.
        This method estimates production on calendar time, conditioned on the
        `time_on` in the future being roughly as it as been in the past.
        Answers: "What will production be if the time is roughly as it has been?"

        >>> x, y, w = well.copy(preprocessing="calendar_time").get_curve_data()
        >>> x
        array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        >>> y
        array([32.,  8.,  8.,  1.,  2.,  1.])
        >>> w
        array([1., 1., 1., 1., 1., 1.])

        Preprocess on producing time.
        Answers: "What will production be if the well is always on the future?"

        >>> x, y, w = well.copy(preprocessing="producing_time").get_curve_data()
        >>> x
        array([0.5  , 1.25 , 2.   , 2.625, 3.25 , 4.25 ])
        >>> y
        array([32., 16.,  8.,  4.,  2.,  1.])
        >>> w
        array([1.  , 0.5 , 1.  , 0.25, 1.  , 1.  ])

        """
        if self.preprocessing == "calendar_time":
            # Shift by 0.5 since data in a time period arrives at the end
            self.period_indexer_ = PeriodIndexer().fit(self.time, time_on=None)
            x = self.period_indexer_.transform(self.time)
            y = self.production.copy()
            w = np.ones_like(self.time_on)
            assert len(x) == len(y) == len(w) == len(self)
            return (x, y, w)

        elif self.preprocessing == "producing_time":
            # Here we pass `time_on` to shift the x-axis
            self.period_indexer_ = PeriodIndexer().fit(self.time, time_on=self.time_on)
            x = self.period_indexer_.transform(self.time)

            # Create production rates and weights
            y = self.production / self.time_on
            w = self.time_on.copy()

            assert len(x) == len(y) == len(w) == len(self)
            return (x, y, w)
        else:
            raise ValueError(f"Invalid method: {self.preprocessing}")

    def get_params(self, q=None) -> dict:
        """Return a dictionary with various parameters, mostly from curve fitting.

        Examples
        --------
        >>> import pandas as pd
        >>> well = Well.generate_random(n=100, seed=25, curve_model="arps")
        >>> well = well.fit(split=0.5, half_life=None, prior_strength=1e-4)
        >>> params = well.get_params(q=[0.5])
        >>> pd.DataFrame.from_records([params]).T
                                   0
        well_id                    1
        segment                    1
        first_time_period    2010-03
        total_time_on      72.171472
        avg_time_on         0.899803
        std                 0.369049
        phi                 0.008876
        p                   1.508815
        sigma               0.431596
        theta_1             4.930999
        theta_2             2.757667
        theta_3            -7.970644
        theta_1_P50         4.869724
        theta_2_P50         2.757667
        theta_3_P50        -7.970644
        q_i                 8.787513
        b                   0.000345
        D_i                 0.063462
        q_i_P50             8.265222
        b_P50               0.000345
        D_i_P50             0.063462
        """
        assert self.is_fitted(), "Must fit to ouput curve parameters"
        model = self.curve_model(*self.curve_parameters_)
        q = [] if q is None else list(q)
        assert all(0 < q_i < 1 for q_i in q)

        # Recall that the distribution of production rate y(t) is given by
        # f(t) * exp(N(0, std_lr)), where std_lr is the long-running standard
        # deviation of the AR(1) process: std_lr = sigma_GND / sqrt(1 - phi^2).
        # We assume the long running distribution is normal by the central limit
        # theorem. Above, sigma_GND(p, sigma, phi) is the standard deviation of
        # the generalized normal distrubiton (GND).
        # We have log( f(t) * exp(N(0, std_lr)) ) = log(f(t)) + N(0, std_lr)
        # = log(f(t, theta1, theta2, theta3)) + N(0, std_lr), which has percentiles:
        # = log(f(t, theta1, theta2, theta3 + z * std_lr)).
        theta_1, *rest = self.curve_parameters_

        # Expected case: E[ln (f * exp(N(0, std_)))] = ln(f) + ln(exp(std_**2/2))
        std_lr = self.std_ * np.sqrt(self.avg_time_on_)
        theta_1_transformed = theta_1 + std_lr**2 / 2
        curve_i = self.curve_model(theta_1_transformed, *rest)
        curve_functions = [("", curve_i)]

        # Percentiles
        for q_i in q:
            percentile = str(round(q_i * 100)).rjust(2, "0")
            z_i = sp.stats.norm().ppf(q_i)
            theta_1_transformed = theta_1 + z_i * std_lr
            curve_model = self.curve_model(theta_1_transformed, *rest)
            curve_functions.append((f"_P{percentile}", curve_model))

        # Loop over every curve (expected + percentiles) and output
        theta_params, original_params = {}, {}
        for postfix, curve_model in curve_functions:
            theta_params.update(
                {
                    f"theta_{i}{postfix}": v
                    for (i, v) in enumerate(curve_model.thetas, 1)
                }
            )

            if isinstance(model, Arps):
                # These are the parameters names used in the Pforecast software
                q_i, b, D_i = curve_model.original_parametrization()
                original_params.update(
                    {
                        f"q_i{postfix}": q_i,
                        f"b{postfix}": b,
                        f"D_i{postfix}": D_i,
                    }
                )
            elif isinstance(model, Exponential):
                C, k = curve_model.original_parametrization()
                original_params.update(
                    {
                        f"C{postfix}": C,
                        f"k{postfix}": k,
                    }
                )
            elif isinstance(model, Constant):
                (C,) = curve_model.original_parametrization()
                original_params.update({f"C{postfix}": C})

        # Weights is the total time on
        _, _, weights = self.get_curve_data()

        record = (
            {
                "well_id": self.id,
                "segment": self.segment,
                "first_time_period": self.time.values[0],
                "total_time_on": float(weights.sum()),
                "avg_time_on": float(self.avg_time_on_),
                "std": self.std_,
                "phi": float(self.parameters_["phi"]),
                "p": float(self.parameters_["p"]),
                "sigma": float(self.parameters_["sigma"]),
            }
            | theta_params
            | original_params
        )
        return record

    def _split_and_remove_zero_time_on(self, array, *, idx: int) -> tuple:
        """Return (test, train) of `array` based on split, removing
        time_on ~= 0 entries."""
        assert len(array) == len(self)
        assert isinstance(idx, int)
        arr_train, arr_test = array[:idx], array[idx:]

        (x, y, w) = self.get_curve_data()
        w_train, w_test = w[:idx], w[idx:]

        # Remove zero-weight data (AFTER splitting, since the split index refers
        # to the original data set). Zero weight data is removed since
        # it will not be used for fitting or for evaluation anyway,
        # and since (y == 0) => (w == 0) we also avoid taking log(y) = log(0)
        mask_train = (w_train > 0) & (~np.isclose(w_train, 0))
        mask_test = (w_test > 0) & (~np.isclose(w_test, 0))
        return arr_train[mask_train], arr_test[mask_test]

    def get_train_test(self, split) -> tuple:
        """Return two tuples (x, y, w) for train and test respectively.
        Zero weight data points are removed from the results.

        Examples
        --------
        Generate some data:

        >>> time = pd.period_range(start="2020-01-01", periods=5, freq="D")
        >>> prod = np.array([10, 4, 4, 1, 1])
        >>> time_on = np.array([1, 0.5, 0.5, 0.25, 0.25])
        >>> well = Well(id=1, time=time, production=prod, time_on=time_on,
        ...        preprocessing="producing_time")

        >>> period = pd.Period("2020-01-03")
        >>> (x_train, _, _), (x_test, _, _) = well.get_train_test(period)
        >>> x_train
        array([0.5 , 1.25])
        >>> x_test
        array([1.75 , 2.125, 2.375])

        On producing time, splits are made on cumulative time.

        >>> (_, y_test, _), (_, y_train, _) = well.get_train_test(0.5)
        >>> y_test
        array([10.])
        """

        # Get the index to split on, then get data
        idx = self.get_split_index(split)
        (x, y, w) = self.get_curve_data()  # (time, production, weights)
        assert np.min(x) >= 0

        # Split data
        x_train, x_test = self._split_and_remove_zero_time_on(x, idx=idx)
        y_train, y_test = self._split_and_remove_zero_time_on(y, idx=idx)
        w_train, w_test = self._split_and_remove_zero_time_on(w, idx=idx)

        return ((x_train, y_train, w_train), (x_test, y_test, w_test))

    def squared_log_error_test(self, split):
        """Returns elementwise (log(p_i) - log(tau_i * f(t_i)))^2 on the test set.

        Examples
        --------
        >>> well = Well.generate_random(n=100, seed=42)
        >>> well = well.fit(split=0.5, half_life=None, prior_strength=1e-4)
        >>> float(well.squared_log_error_test(split=0.5).mean())
        0.144041469...
        """
        assert self.is_fitted()

        # Get test set and split index
        _, (x_test, _, w_test) = self.get_train_test(split)

        if np.isclose(np.sum(w_test), 0.0):
            return w_test

        # Expected production, in normal space (not log-space)
        prod_pred = self.predict(x=x_test, time_on=w_test, q=None)

        # Observed production
        idx = self.get_split_index(split)
        _, prod_obs = self._split_and_remove_zero_time_on(self.production, idx=idx)

        assert len(prod_pred) == len(prod_obs)
        return (np.log(prod_pred) - np.log(prod_obs)) ** 2

    def negative_ll_test(self, split, p=None, sigma=None, phi=None):
        """Return an array of negative log likelihhood evaluations on the
        test set. If std=None, then the inferred std is used."""
        assert self.is_fitted()

        # Get test set
        train, test = self.get_train_test(split)
        x_train, y_train, w_train = train
        x_test, y_test, w_test = test

        if np.isclose(np.sum(w_test), 0.0):
            return w_test

        # Compute prev-eta based on the last training data point
        if len(y_train) > 0:
            curve = self.curve_model(*self.curve_parameters_)
            prev_eta = np.log(y_train[-1]) - curve.eval_log(x_train[-1])
        else:
            prev_eta = None

        # Create a model and return elementwise negative log-likelihood
        model = AR1Model(curve_cls=self.curve_model)
        return model.negative_ll(
            # Parameters (None => use fitted values)
            theta=self.curve_parameters_,
            sigma=self.parameters_["sigma"] if sigma is None else sigma,
            phi=self.parameters_["phi"] if phi is None else phi,
            p=self.parameters_["p"] if p is None else p,
            # Data
            tau=w_test,
            t=x_test,
            y=y_test,
            # Metadata
            half_life=np.inf,
            prev_eta=prev_eta,
        )

    def score(self, split, *, p=None, sigma=None, phi=None):
        """Score the model on a test set defined by the split. Lower is better."""
        neg_ll = self.negative_ll_test(split=split, p=p, sigma=sigma, phi=phi)
        return float(np.sum(neg_ll))

    def forecasting_grid(self, periods: int, return_periods=False):
        """Create foreasting grid for a specified number of periods.

        If `return_periods` is True, then returns (x_grid, periods) instead of
        just the x_grid.

        Examples
        --------

        With `preprocessing="calendar_time"`:

        >>> well = Well(id=1,
        ...    time=pd.period_range(start="2020-01-01", periods=6, freq="D"),
        ...    production=np.array([1000, 500, 65, 125, 65, 30]),
        ...    time_on=np.array([1, 1, 0.25, 0.5, 0.5, 0.5]),
        ...    preprocessing='calendar_time')
        >>> well = well.fit(p=2, half_life=2, prior_strength=1)
        >>> well.forecasting_grid(3)
        array([6.5, 7.5, 8.5])
        >>> x_grid, periods_grid = well.forecasting_grid(3, return_periods=True)
        >>> well.time[-1]
        Period('2020-01-06', 'D')
        >>> periods_grid
        PeriodIndex(['2020-01-07', '2020-01-08', '2020-01-09'], dtype='period[D]')

        With `preprocessing="producing_time"`:

        >>> well = Well.from_dirty_data(id=1,
        ...    time=pd.period_range(start="2020-01-01", periods=6, freq="D"),
        ...    production=np.array([1000, 500, 0, 125, 65, 30]),
        ...    time_on=np.array([1, 1, 0, 0.5, 0.5, 0.5]),
        ...    preprocessing='producing_time')
        >>> well = well.fit(p=2, half_life=2, prior_strength=1)
        >>> well.forecasting_grid(3)
        array([4., 5., 6.])
        >>> x_grid, periods_grid = well.forecasting_grid(3, return_periods=True)
        >>> well.time[-1]
        Period('2020-01-06', 'D')
        >>> periods_grid
        PeriodIndex(['2020-01-07', '2020-01-08', '2020-01-09'], dtype='period[D]')

        """
        assert (self.time.sort_values().values == self.time.values).all(), (
            "Must be sorted"
        )
        assert isinstance(periods, int), f"Expected int, got {periods}, {type(periods)}"

        # Create a grid with future Periods
        last_period = self.time.values[-1]
        future_periods = (
            pd.period_range(start=last_period, periods=periods, freq=last_period.freq)
            + 1
        )
        assert future_periods.freq == self.time.values[0].freq

        # Transform from future Periods to future x-values to evaluate on
        future_grid = self.period_indexer_.transform(future_periods)
        (x, y, w) = self.get_curve_data()
        assert np.min(future_grid) > np.max(x), "Grid point must be in the future"

        if return_periods:
            return future_grid, future_periods

        return future_grid

    def predict(self, x, *, time_on, q=None):
        """Given an array x of times, evaluate the model (not in log-space).
        Returns the estimated production rate y-hat, taking uncertainty into
        account (unless q=0.5). The caller has to multiply the returned values
        by time_on to get predicted production.

        Values of `time_on` used here should roughly match those seen in
        training data, since that is the resolution at which we assume
        log-normal errors.

        NOTE: For non-Normal errors, this method returns an approximation since
        when we exponentiate the distribution, we assume that is is Normal.
        In reality, we allow for generalized normal distributions.
        Simulation might lead to a more accurate result.

        If q=None, then the expected curve will be returned. This computation
        takes uncertainty into account. The formulas is:
            E[y_i] = E[exp(log(y_i))] = E[exp(log(f_i) + N(0, std_i))]
                 = E[f_i * exp(N(0, std_i))] = f * E[exp(N(0, std_i))]
                 = f_i * exp(0 + std_i^2/2)
        Where std_i = std * sqrt(time_on).
        See https://gregorygundersen.com/blog/2022/05/24/square-root-of-time-rule/
        and https://en.wikipedia.org/wiki/Log-normal_distribution for more info.

        If q is not None, then the curve is shifted up to down to that quantile.
        Working in log-space, the formula is:
            F^-1(p) = mu + std * z_p = log(f) + std * z_p
                    = log( arps(theta1, ...) ) + std * z_p
                    = log( arps(theta1 + std * z_p, ...))
        where z_p = scipy.stats.norm().ppf(q). This value is then exponentiated.
        Note that if q=0.5, the result is equivalent to E[y] with no uncertainty.

        Examples
        --------
        >>> n = 999
        >>> rng = np.random.default_rng(42)
        >>> time_on = np.ones(n, dtype=float) * 0.1
        >>> noise = rng.normal(loc=0, scale=1, size=n)
        >>> production = time_on * np.exp(noise)

        Create a Well and fit it.

        >>> time = pd.period_range(start="2020-01-01", periods=n, freq="D")
        >>> well = Well(production=production, time_on=time_on, time=time,
        ...             curve_model="constant", preprocessing="producing_time")
        >>> well = well.fit(p=2.0, phi=0, half_life=None, prior_strength=0, split=1.0)

        Predict P50 (median) production rates:

        >>> x = np.array([0, 1, 2])
        >>> time_on = np.ones_like(x) * 0.1
        >>> float(well.predict(x, time_on=time_on, q=0.5)[0])
        0.9704...

        Predict expected production rates, with noise accounted for:

        >>> float(well.predict(x, time_on=time_on, q=None)[0])
        1.5538...

        Finally, predict P10 and P90:

        >>> float(well.predict(x, time_on=time_on, q=0.1)[0])
        0.2798...
        >>> float(well.predict(x, time_on=time_on, q=0.9)[0])
        3.3651...

        """
        assert self.is_fitted()
        assert q is None or (0 < q < 1), "q must be None or 0 < q < 1"
        assert np.allclose(np.sort(x), x), "grid `x` must be sorted"
        assert len(time_on) == len(x)
        assert np.all((np.min(time_on) > -1e-6) & (np.max(time_on) < 1 + 1e-6))

        # `standard_deviations` != 0, so we need to shift the curve.
        curve = self.curve_model(*self.curve_parameters_)
        std_i = self.std_ * np.sqrt(time_on)
        if q is None:
            # Expected value E[f * exp(N(0, std))] = f * exp(sigma^2/2)
            # https://en.wikipedia.org/wiki/Log-normal_distribution
            return curve.eval(x) * np.exp(std_i**2 / 2)
        else:
            z = sp.stats.norm().ppf(q)
            return curve.eval(x) * np.exp(z * std_i)

    def simulate_cumprod(self, x, *, prev_eta=None, seed=None, simulations=999):
        """Simulate cumulative production, starting at the first x-value.

        - self.avg_time_on_ = tau is used as the spacing for simulations.
          For instance, if x = [2, 3, 4] is the grid and tau = 0.428, then
          the simulation grid chosen is [2, 2.428, 2.857, 3.285, ...]
        - The simulation is computed as:
            cumsum_i tau * f(x_i) * exp(N(0, sqrt(tau) * std))
        - The result is interpolate back onto the original x-grid.

        The returned numpy array has shape (simulations, len(x)).

        Examples
        --------
        >>> n = 999
        >>> rng = np.random.default_rng(42)
        >>> time_on = np.ones(n, dtype=float) * 1.0
        >>> noise = rng.normal(loc=0, scale=1, size=n)
        >>> production = time_on * np.exp(noise)

        Create a Well and fit it.

        >>> time = pd.period_range(start="2020-01-01", periods=n, freq="D")
        >>> well = Well(production=production, time_on=time_on, time=time,
        ...             curve_model="constant", preprocessing="producing_time")
        >>> well = well.fit(p=2.0, half_life=None, prior_strength=0, split=1.0)

        Simulate cumulative production:

        >>> x = np.array([1, 2, 3, 4, 5])
        >>> sims = well.simulate_cumprod(x, seed=0, simulations=999)
        >>> np.mean(sims, axis=0)
        array([1.67..., 3.29..., 4.85..., 6.42..., 8.02...])
        """
        assert isinstance(simulations, int) and simulations > 0
        assert seed is None or isinstance(seed, int)
        assert self.is_fitted()
        assert np.allclose(np.sort(x), x), "grid `x` must be sorted"
        tau = self.avg_time_on_

        # Create an equidistant grid with spacing `avg_time_on_` to simulate on
        x_fine = np.arange(start=np.min(x), stop=np.max(x) + tau, step=tau)
        assert np.min(x_fine) <= np.min(x)
        assert np.max(x_fine) >= np.max(x)

        # Simulate production rates
        time_on = np.ones_like(x_fine) * tau
        sims = self.simulate(
            x=x_fine,
            time_on=time_on,
            seed=seed,
            prev_eta=prev_eta,
            simulations=simulations,
        )

        # The production is the production rate times the time elapsed.
        cumsums = np.cumsum(tau * sims, axis=1)

        # Interpolate back onto the original x grid, then return
        return np.array([np.interp(x, x_fine, cumsum) for cumsum in cumsums])

    def simulate(self, x, *, time_on=None, prev_eta=None, seed=None, simulations=999):
        """Return simulations of production rates on the x-grid. The model is:

          exp(log(f(t, theta)) + epsilon) = f(t, theta) * exp(epsilon)

        where epsilon ~ N(0, sqrt(std) * time_on) and the std is estimated
        during curve fitting. Multiply by the returned result by time_on to
        get predicted production.

        The returned numpy array has shape (simulations, len(x)).
        """
        assert isinstance(simulations, int) and simulations > 0
        assert seed is None or isinstance(seed, int)
        assert self.is_fitted()
        assert np.allclose(np.sort(x), x), "grid `x` must be sorted"
        if time_on is None:
            x_diffs = np.diff(x)
            if x_diffs.size and not np.allclose(x_diffs, x_diffs[0]):
                raise ValueError("The grid `x` must be sorted and equidistant.")
            time_on = np.ones_like(x, dtype=float)
        assert len(time_on) == len(x)
        assert np.all((np.min(time_on) > -1e-6) & (np.max(time_on) < 1 + 1e-6))

        # Create model, set parameters as learned
        model = AR1Model(curve_cls=self.curve_model)
        model.update(**self.parameters_)

        # Simulate production rates y
        prod_rates = model.simulate(
            tau=time_on, t=x, seed=seed, prev_eta=prev_eta, simulations=simulations
        )
        return prod_rates

    def __len__(self):
        return len(self.time)

    def copy(self, **kwargs):
        new = copy.deepcopy(self)
        new.__dict__.update(**kwargs)
        return new

    def split(self, segments=None):
        """Given a list of tuples [(start_eriod, end_period), ...],
        split the Well into segments and yield new Well instances."""
        assert self.segment == 1

        # If there are no segments, yield the Well with one segment
        if segments is None:
            yield self.copy()
            return

        # If there are segments, loop over them in order
        for segment_number, (start, end) in enumerate(sorted(segments), 1):
            assert isinstance(start, pd.Period) or pd.isna(start)
            assert isinstance(end, pd.Period) or pd.isna(end)

            # Split out the segment using two masks for start and end
            ones = np.ones(len(self.time), dtype=bool)
            start = ones if pd.isna(start) else (self.time >= start)
            end = ones if pd.isna(end) else (self.time < end)
            mask = start & end

            # Copy the object and mask out the data
            well = self.copy()
            well.segment = segment_number
            well.production = well.production[mask]
            well.time_on = well.time_on[mask]
            well.time = well.time[mask]

            yield well

    @classmethod
    def generate_random(
        cls,
        n=None,
        *,
        curve_model=None,
        curve_parameters=None,
        time_on=None,
        std=0.33,
        seed=None,
        **kwargs,
    ):
        """Generate a random well.

        Examples
        --------
        >>> well = Well.generate_random(6, seed=0, preprocessing="producing_time")
        >>> (x, y, w), _ = well.get_train_test(split=1.0)
        >>> y
        array([3.8455317 , 1.64007649, 0.79042216, 0.13533661, 0.08540875])
        >>> w
        array([0.95589751, 0.87720693, 0.72652461, 0.97954296, 0.99091283])

        It's possibly to supply curve parameters:

        >>> Well.generate_random(n=25, curve_model="arps", curve_parameters=(0, 0, 0))
        Well(id='1', segment=1)

        Any keyword argument is passed:

        >>> Well.generate_random(id="testwell", segment=2, time_on=[1, 1])
        Well(id='testwell', segment=2)
        """
        rng = np.random.default_rng(seed)

        # Logic for n and time_on
        if isinstance(n, int) and time_on is None:
            time_on = rng.random(size=n) ** 0.1
            num_shutoff = n // 5

            if num_shutoff > 0:
                shutoff_idx = rng.choice(np.arange(n), size=num_shutoff, replace=False)
                time_on[shutoff_idx] = 0.0

        elif (n is None) and (isinstance(time_on, (list, tuple, np.ndarray))):
            n = len(time_on)
            time_on = np.array(time_on)
        elif n is None and time_on is None:
            return cls.generate_random(
                n=25,
                curve_model=curve_model,
                curve_parameters=curve_parameters,
                time_on=time_on,
                std=std,
                seed=seed,
                **kwargs,
            )

        else:
            raise Exception("Error with args `n` and `time_on`")

        # Logic for curve model
        if curve_model is None:
            models = [(k, v) for (k, v) in cls.model_map.items() if k != "constant"]
            curve_model, curve_cls = rng.choice(models)
        else:
            curve_cls = cls.model_map[curve_model]

        if curve_parameters is None and curve_model == "exponential":
            C = rng.normal(10, 0.1)  # Start value
            end = rng.normal(0.01, 0.001)  # Value after n periods
            k = (np.log(C) - np.log(end)) / n
            f = curve_cls.from_original_parametrization(C, k)
        elif curve_parameters is None and curve_model == "arps":
            q_1 = rng.normal(10, 0.1)
            end = rng.normal(0.01, 0.001)
            h = (
                rng.triangular(0, 0, 1) ** 2
            )  # h in range (0, 1), biased towards 0 (exp)
            D = ((q_1 / end) ** h - 1) / (h * n)
            f = curve_cls.from_original_parametrization(q_1, h, D)
        elif curve_parameters is None and curve_model == "constant":
            C = rng.normal(10)  # Start value
            f = curve_cls.from_original_parametrization(C)
        else:
            f = curve_cls(*curve_parameters)

        assert isinstance(f, tuple(cls.model_map.values()))

        # Integrate the curve to obtain production for each time period
        # \int_T1^T2 f(t) dt ~= f((T2 - T1) / 2) * (T2 - T1)
        cumulative_time_on = np.cumsum(time_on)
        midpoints = cumulative_time_on - time_on / 2
        production = f.eval(midpoints) * time_on
        production[np.isclose(time_on, 0)] = 0.0

        # Production must be positive, no time implies no production
        assert np.all(production >= 0)
        assert np.allclose(np.sum(production[time_on == 0.0]), 0)

        # Add noise
        m = ~np.isclose(production, 0)  # Positive production
        epsilon = rng.normal(scale=std, size=m.sum())
        # Here we correct the standard deviation with sqrt(n) since the sum
        # of n variables has standard deviation sqrt(n)
        production[m] = np.exp(np.log(production[m]) + np.sqrt(time_on[m]) * epsilon)

        # Create time
        freq = str(rng.choice(["D", "M"]))
        freq = kwargs.pop("freq", freq)
        time = pd.period_range(start="2010-01-01", periods=n, freq=freq)

        # Create preprocessing
        preprocessing = rng.choice(cls.preprocessing_options)

        id = kwargs.pop("id", 1)
        time = kwargs.pop("time", time)
        preprocessing = kwargs.pop("preprocessing", preprocessing)
        curve_model = kwargs.pop("curve_model", curve_model)

        return cls.from_dirty_data(
            id=id,
            production=production,
            time=time,
            time_on=time_on,
            preprocessing=preprocessing,
            curve_model=curve_model,
            **kwargs,
        )


class WellGroup(UserList):
    """A group of wells."""

    def tune_hyperparameters(
        self,
        *,
        # hyperparameters
        half_life: float,
        prior_strength: float,
        # Default is to fit on all data
        split=0.5,
        neg_logpdf_prior=None,
        # Number of function evaluations used when hyperparameter tuning
        hyperparam_maxfun=100,
        # Possibly fixed parameters
        p=None,
        sigma=None,
        phi=None,
    ):
        """Tune hyperparameters that are given as intervals.
        Hyperparameters NOT given as intervals are fixed."""

        log.info("Tuning hyperparameters.")

        # Set up search bounds for optimization routine
        parameter_names = ["half_life", "prior_strength"]
        parameter_values = [half_life, prior_strength]

        # Split parameters into free (optimize over) and fized (do not optimize)
        names_vals = list(zip(parameter_names, parameter_values))
        params_free = {n: v for (n, v) in names_vals if isinstance(v, list)}
        params_fixed = {n: v for (n, v) in names_vals if not isinstance(v, list)}

        log.info(f"Hyperparam tuning. Fixed parameters: {params_fixed}")
        log.info(f"Hyperparam tuning. Free parameters: {params_free}")

        # All parameters are fixed -> nothing to optimize over
        if not params_free:
            return params_fixed

        # Names and values of free parameters
        names_free = list(params_free.keys())
        bounds_free = list(params_free.values())

        def func(parameters):
            """Function called by optimizer."""

            # Transform the free parameters from sampling space to actual space
            free_args = transform_parameters(dict(zip(names_free, parameters)))

            # Pass both bound and free parameters. Fit, then evaluate.
            self.fit(
                split=split,
                neg_logpdf_prior=neg_logpdf_prior,
                p=p,
                sigma=sigma,
                phi=phi,
                **params_fixed,
                **free_args,
            )

            # TODO: It is not entirely clear that hyperparameter tuning should
            # use the negative ll as the metric. We are free to use RMSE, RMSE
            # in logspace, RMSE of cumulative at the final point in time on
            # the test set, or negative log PDF of a normal at the final point
            # in time on the test set (assuming normality in logspace or
            # original space). For now we stick to the simplest: evaluate by
            # neative log loss, and also report RMSE.
            log_loss = self.score(
                split,
                p=p,
                sigma=sigma,
                phi=phi,
            )
            rmse = self.rmse_log(split)
            relative_error_expected, relative_error_P50 = list(
                self.relative_cumulative_error(split, q=[None, 0.5])
            )

            # Information for logging
            param_trans = transform_parameters(dict(zip(names_free, parameters)))
            params = {k: float(round(v, 4)) for (k, v) in param_trans.items()}

            log.info(f"{params}")
            log.info(f"  Negative log-likelihood: {fnum(log_loss)}")
            log.info(f"  RMSE in logspace:         {fnum(rmse)}")
            log.info(f"  Rel. error (expected):    {relative_error_expected:.2%}")
            log.info(f"  Rel. error (P50):        {relative_error_P50:.2%}")

            return log_loss

        # Use DIviding RECTangles (DIRECT) optimizer. This typically produces
        # better results than Nelder-Mead and random sampling.
        result = sp.optimize.direct(
            func=func,
            bounds=list(transform_bounds(names_free, bounds_free)),
            maxfun=hyperparam_maxfun,
        )
        result_x = tuple(float(x_i) for x_i in result.x)

        # Transform from sampling space to actual space, then return
        params_free = transform_parameters(dict(zip(names_free, result_x)))
        return {**params_free, **params_fixed}

    def fit(
        self,
        *,
        # hyperparameters
        half_life: float,
        prior_strength: float,
        # parameters
        p=None,  # p-norm, or beta in generalized normal distribution (GND)
        sigma=None,  # scale parameter for GND (larger than 0)
        phi=None,  # autocorrelation parameter (between 0 and 1)
        # Default is to fit on all data
        split=1.0,
        neg_logpdf_prior=None,
    ) -> None:
        """Fit every Well."""

        return type(self)(
            well.fit(
                # hyperparams
                half_life=half_life,
                prior_strength=prior_strength,
                # params
                p=p,
                sigma=sigma,
                phi=phi,
                # other
                split=split,
                neg_logpdf_prior=neg_logpdf_prior,
            )
            for well in self
        )

    def to_df(self, forecast_periods: int = 0, q=None, simulations=999) -> pd.DataFrame:
        """Create a DataFrame with every well."""
        return pd.concat(
            [
                well.to_df(
                    forecast_periods=forecast_periods, q=q, simulations=simulations
                )
                for well in self
            ],
            axis=0,
            ignore_index=True,
        )

    def score(self, split, *, p=None, sigma=None, phi=None) -> float:
        """Score all wells on the test set weighted by `time_on`."""
        # Get a list of numpy arrays
        ll = [w.negative_ll_test(split=split, p=p, sigma=sigma, phi=phi) for w in self]

        # Concatenate all test log-likelihoods together
        ll = np.concatenate(ll)

        # On WellGroup we are more conservative than on a single Well.
        # On Well we return if test set is empty, but if ALL test sets are empty we fail:
        if len(ll) < 1:
            raise Exception(
                f"Cannot evaluate on test-set because it was empty. Split={split}"
            )

        # Unpack to one numpy array and take average
        # This implies that wells with fewer data points get weighted down.
        # An alternative would be to do a mean-of-means
        return float(np.mean(ll))

    def rmse_log(self, split) -> float:
        """Compute RMSE in log-space on all wells on the test set,
        weighted by `time_on`."""
        # Get a list of numpy arrays
        squared_errors = [w.squared_log_error_test(split=split) for w in self]

        # Unpack w_test from ((x_train, y_train, w_train), (x_test, y_test, w_test))
        weights = [w.get_train_test(split)[1][2] for w in self]

        squared_errors = np.concatenate(squared_errors)
        weights = np.concatenate(weights)
        assert len(squared_errors) == len(weights)
        # On WellGroup we are more conservative than on a single Well.
        # On Well we return if test set is empty, but if ALL test sets are empty we fail:
        if len(weights) < 1:
            raise Exception(
                f"Cannot evaluate on test-set because it was empty. Split={split}"
            )

        return float(np.sqrt(np.average(squared_errors, weights=weights)))

    def get_params(self, q=None) -> pd.DataFrame:
        """Create a DataFrame for each well record."""
        records = [well.get_params(q=q) for well in self]
        return pd.DataFrame.from_records(records)

    def aggregate(self, group_ids):
        """Returns a new WellGroup with summed well information.

        - The IDs of Wells to sum must be unique
        - There can only be one segment (with integer 1) in wells to sum
        - The new, summed well will be placed at the end in the WellGroup
        - IDs will be concatenated with a comma. e.g. ['A', 'B'] => 'A,B'

        Examples
        --------
        >>> kwargs = {"curve_model": "arps", "preprocessing": "producing_time", "freq":"D"}
        >>> w1 = Well.generate_random(n=100, id='1', **kwargs)
        >>> w2 = Well.generate_random(n=50,  id='2', **kwargs)
        >>> w3 = Well.generate_random(n=25,  id='3', **kwargs)
        >>> WellGroup([w1, w2, w3]).aggregate(['3', '1'])
        [Well(id='2', segment=1), Well(id='3,1', segment=1)]
        """

        # IDs are converted to string in Well, so convert here too
        group_ids = [str(well_id) for well_id in group_ids]

        # Check that there is exactly one Well with each ID to sum
        id_counts = collections.Counter(well.id for well in self)
        for well_id in group_ids:
            if id_counts.get(well_id, None) != 1:
                raise ValueError(f"Need exactly one well with id {well_id}")
            if WELLGROUP_SEP in well_id:
                raise ValueError(f"Cannot aggregate twice: {WELLGROUP_SEP}")

        # Select wells to sum and wells to not sum. Sorted to maintain order w.r.t input
        wells_to_sum = sorted(
            [well for well in self if well.id in group_ids],
            key=lambda w: group_ids.index(w.id),
        )
        wells_to_not_sum = [well for well in self if well.id not in group_ids]

        # Validate data
        for w1, w2 in pairwise(wells_to_sum):
            assert w1.segment == w2.segment == 1, "All segments must be 1"
            assert w1.preprocessing == w2.preprocessing, "'preprocessing' must match"
            assert w1.curve_model == w2.curve_model, "'curve_model' must match"
            assert not w1.is_fitted(), "Cannot sum fitted wells"
            assert not w2.is_fitted(), "Cannot sum fitted wells"

        # Vectorize sums over each period
        production_series = [
            pd.Series(w.production, index=w.time) for w in wells_to_sum
        ]
        prodrate_series = [
            pd.Series(w.production / w.time_on, index=w.time) for w in wells_to_sum
        ]

        # The sum of production is straightfoward, simply sum within each period
        # and make sure that NaN (no data) is equal to a production value of 0.
        add_func = functools.partial(pd.Series.add, fill_value=0)
        sum_production = functools.reduce(add_func, production_series)

        # To aggregate time_on properly, observe that the propery that we want
        # to obey is:
        #     p1 / t2 + p2 / t2 + p3 / t3 = P / T
        # where P is the aggregated production (the sum) and T is the aggregated
        # time_on. Solve this equation for T to find that T should be a weighted
        # harmonic mean:
        #     T = (sum_i p_i) / (sum_i p_i / t_i)
        sum_prodrate = functools.reduce(add_func, prodrate_series)

        # Aggregated values (sum and weighted harmonic sum)
        agg_time = sum_production.index
        agg_production = sum_production.to_numpy()
        agg_time_on = (sum_production / sum_prodrate).to_numpy()

        # Create the new summed well
        summed_well = Well(
            production=agg_production,
            time=agg_time,
            time_on=agg_time_on,
            id=WELLGROUP_SEP.join(w.id for w in wells_to_sum),
            segment=1,  # Wells can only be summed if segment=1
            preprocessing=wells_to_sum[0].preprocessing,
            curve_model=wells_to_sum[0].curve_model_str_,
        )

        return WellGroup(wells_to_not_sum + [summed_well])

    def relative_cumulative_error(self, split, q=None):
        """Compute relative error over cumulative forecast, i.e.,

        (sum_forecast - sum_actual) / sum_actual

        """
        # Get (sum_forecast, sum_actual) for each well and method, e.g.
        # [[(well1_exp_forecast, well1_exp_actual), (well1_P50_forecast, well1_P50_actual)], ...]
        results = [
            list(well.relative_cumulative_error(split=split, q=q)) for well in self
        ]

        # Loop over every method in each q_i in q, e.g. expected, P50, ...
        for i in range(len(results[0])):
            sum_forecast = np.sum([result[i][0] for result in results])
            sum_actual = np.sum([result[i][1] for result in results])
            yield float((sum_forecast - sum_actual) / sum_actual)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys", "-x"])
