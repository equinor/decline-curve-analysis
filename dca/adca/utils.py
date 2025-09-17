"""
Utility functions and classes that do not fit any other place.
"""

import copy
import functools
import itertools
import logging
import re

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp

from dca.timeseries import TimeSeriesInterpolator

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def to_filename(wellnames):
    """Convert ID to filenames.

    Examples
    --------
    >>> wellnames = ["NO 31/6-A-30", "NO 99/2-W-12 H", "NO 99/2-B-12 H"]
    >>> to_filename(wellnames)
    'NO 31/6-A-30,NO 99/2-W-12 H,1more'
    >>> to_filename(wellnames[:2])
    'NO 31/6-A-30,NO 99/2-W-12 H'
    """
    if len(wellnames) > 2:
        return ",".join(wellnames[:2]) + f",{len(wellnames) - 2}more"
    else:
        return ",".join(wellnames)


def to_period(value, freq):
    """Converts a value like '2020-01-25' to a pd.Period.

    This function is extra security, since Pandas allows:

        pd.Period('2020-01-01', freq='M') => Period('2020-01', 'M')
        pd.Period('2020-01', freq='D')    => Period('2020-01-01', 'D')
        pd.Period('01-01-2020', freq='D') => Period('2020-01-01', 'D')

    Examples
    --------
    >>> to_period('2020-01-25', freq='D')
    Period('2020-01-25', 'D')
    >>> to_period('2020-01', freq='D')
    Traceback (most recent call last):
    ...
    ValueError: Period 'value='2020-01'' incompatible with 'freq='D''
    """

    # Matches pattern like YYYY-MM and YYYY-MM-DD
    pattern_month = r"^\d{4}-(0[1-9]|1[0-2])$"
    pattern_day = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$"

    if value is None:
        return pd.Period(value=value, freq=freq)
    elif re.match(pattern_month, value) and freq == "M":
        return pd.Period(value=value, freq=freq)
    elif re.match(pattern_day, value) and freq == "D":
        return pd.Period(value=value, freq=freq)
    else:
        raise ValueError(f"Period '{value=}' incompatible with '{freq=}'")


class PeriodIndexer:
    """Convert a sequence of Periods to numbers for curve fitting."""

    def fit(self, periods, time_on=None):
        """Fit to data.

        Examples
        --------
        >>> import pandas as pd
        >>> periods = pd.period_range(start="2020-01-01", periods=4, freq="D")
        >>> period_indexer = PeriodIndexer().fit(periods)
        >>> prange = pd.period_range(start="2020-01-01", periods=4, freq="D")
        >>> periods = pd.Series(prange)
        >>> period_indexer = PeriodIndexer().fit(periods)
        """
        assert isinstance(periods.dtype, pd.PeriodDtype)
        assert len(periods) > 0
        assert isinstance(time_on, np.ndarray) or (time_on is None)
        assert periods.is_monotonic_increasing
        assert (time_on is None) or np.all(np.isfinite(time_on))
        assert (time_on is None) or np.min(time_on) >= 0
        assert (time_on is None) or np.max(time_on) <= 1

        self.periods = periods.copy()

        # Copy if it's an array
        self.time_on = time_on.copy() if time_on is not None else time_on

        # If time_on is not None, compute the cumulative time
        if time_on is not None:
            self.cumulative_time_on = np.cumsum(time_on)
            self.midpoints = self.cumulative_time_on - time_on

        return self

    def transform(self, periods):
        """Transform to floats for curve fitting.

        Examples
        --------

        Examples where `time_on` is not given:

        >>> periods = pd.period_range(start="2020-01-01", periods=4, freq="D")
        >>> PeriodIndexer().fit_transform(periods)
        array([0.5, 1.5, 2.5, 3.5])
        >>> months = pd.period_range(start="2020-01", periods=4, freq="M")
        >>> p = PeriodIndexer().fit(months)
        >>> months_extrapolate = pd.Series([pd.Period("2020-02", freq="M"),
        ...                                 pd.Period("2020-08", freq="M")])
        >>> p.transform(months_extrapolate)
        array([1.5, 7.5])

        With a non-equidistant sequence of periods:

        >>> periods = pd.period_range(start="2020-01-01", periods=9, freq="D")
        >>> periods_fit = periods[[0, 1, 3, 4]]
        >>> periods_transform = periods[[0, 1, 3, 4, 6, 8]]
        >>> p = PeriodIndexer().fit(periods_fit)
        >>> p.transform(periods_fit)
        array([0.5, 1.5, 3.5, 4.5])
        >>> p.transform(periods_transform)
        array([0.5, 1.5, 3.5, 4.5, 6.5, 8.5])

        Examples where `time_on` is given.

        >>> periods = pd.period_range(start="2020-01-01", periods=4, freq="D")
        >>> time_on = np.array([0.5, 0.5, 0, 1])
        >>> indexer = PeriodIndexer().fit(periods, time_on=time_on)
        >>> indexer.transform(periods)
        array([0.25, 0.75, 1.  , 1.5 ])
        >>> periods_future = pd.period_range(start="2020-01-01", periods=8, freq="D")
        >>> indexer.transform(periods_future)
        array([0.25, 0.75, 1.  , 1.5 , 2.5 , 3.5 , 4.5 , 5.5 ])

        With a non-equidistant sequence of periods:

        >>> periods = pd.period_range(start="2020-01-01", periods=9, freq="D")
        >>> periods_fit = periods[[0, 2, 4, 6]]
        >>> time_on_fit = np.array([0.5, 0.5, 0, 0.8])
        >>> periods_transform = periods[[0, 1, 2, 3, 7, 8]]
        >>> p = PeriodIndexer().fit(periods_fit, time_on=time_on_fit)
        >>> p.transform(periods_fit)
        array([0.25, 0.75, 1.  , 1.4 ])
        >>> p.transform(periods_transform)
        array([0.25, 0.5 , 0.75, 1.  , 2.3 , 3.3 ])

        Another test case:

        >>> periods = pd.period_range(start="2020-01-01", periods=6, freq="D")
        >>> time_on = np.array([1, 1, 0.5, 0, 1, 0])
        >>> PeriodIndexer().fit_transform(periods, time_on=time_on)
        array([0.5 , 1.5 , 2.25, 2.5 , 3.  , 3.5 ])

        And another:

        >>> periods = pd.period_range(start="2020-01-01", periods=6, freq="D")
        >>> time_on = np.array([1, 1, 0, 0.5, 0.5, 0.5])
        >>> PeriodIndexer().fit_transform(periods, time_on=time_on)
        array([0.5 , 1.5 , 2.  , 2.25, 2.75, 3.25])

        """
        assert periods.is_monotonic_increasing

        # Get information from the periods that we fitted on
        periods_min = self.periods.min()
        periods_max = self.periods.max()

        # --> SIMPLE CASE: no time_on was passed, so assume it's 1 always and
        # simply count how many periods from the start time have elapsed
        if self.time_on is None:
            # Create a pd.Index with pd.offsets.Day
            offsets = periods - periods_min
            return np.array([offset.n + 0.5 for offset in offsets])

        # --> HARD CASE: time_on was passed, so we need to look at the cumulative
        # time on, and also deal with the case where we predict on the future.

        # Get time_on for each period that we want to transform
        time_on = []
        for period in periods:
            assert period >= periods_min, "New periods cannot be before old ones"

            # Find the cumulative time on at the current period
            i = self.periods.searchsorted(period, side="right") - 1
            time = self.cumulative_time_on[i]
            if period in self.periods:
                time -= self.time_on[i] / 2

            # If the current period is in the future, add ones for each period
            if i == len(self.periods) - 1 and period != periods_max:
                time += 0.5 + ((period - periods_max).n - 1)

            time_on.append(time)

        return np.array(time_on)

    def fit_transform(self, periods, time_on=None):
        """Fit and transform."""
        return self.fit(periods, time_on=time_on).transform(periods)


class Bunch(dict):
    """Container object exposing keys as attributes.

    Bunch objects are sometimes used as an output for functions and methods.
    They extend dictionaries by enabling values to be accessed by key,
    `bunch["value_key"]`, or by an attribute, `bunch.value_key`.

    Examples
    --------
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    >>> b = Bunch(**{"a":{"c":2}})
    >>> b.a.c
    2
    """

    # https://github.com/scikit-learn/scikit-learn/blob/f07e0138b/sklearn/utils/_bunch.py
    # https://stackoverflow.com/questions/38034377/object-like-attribute-access-for-nested-dictionary

    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """Construct nested AttrDicts from nested dictionaries."""
            if not isinstance(data, dict):
                return data
            else:
                return type(self)({key: from_nested_dict(data[key]) for key in data})

        super().__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])


def transform_parameters(parameters: dict) -> dict:
    """Transform from sampling space to actual space.

    Examples
    --------
    >>> transform_parameters({'half_life': 3})
    {'half_life': 1000}
    >>> transform_parameters({'prior_strength': 0, 'p':1.5})
    {'prior_strength': 1, 'p': 1.5}
    """

    # Transform the parameters from log-scale to actual scale.
    # This is done since we want to sample e.g. half_life on logscale
    parameters = parameters.copy()
    valid_names = {"p", "half_life", "prior_strength"}
    assert all((key in valid_names) for key in parameters.keys())

    if "half_life" in parameters.keys():
        parameters["half_life"] = 10 ** parameters["half_life"]

    if "prior_strength" in parameters.keys():
        parameters["prior_strength"] = 10 ** parameters["prior_strength"]

    return parameters


def transform_bounds(parameter_names: list[str], parameter_bounds: list[list]):
    """Yields transformed parameter bounds from actual space to sampling space.

    Examples
    --------
    >>> list(transform_bounds(["half_life"], [[10, 1000]]))
    [(1.0, 3.0)]
    """
    assert all(isinstance(bound, list) for bound in parameter_bounds)
    valid_names = {"p", "half_life", "prior_strength"}
    assert all((name in valid_names) for name in parameter_names)
    assert len(parameter_names) == len(parameter_bounds)

    for name, (lower_bound, upper_bound) in zip(parameter_names, parameter_bounds):
        if name in ("half_life", "prior_strength"):
            yield (float(np.log10(lower_bound)), float(np.log10(upper_bound)))
        else:
            yield (lower_bound, upper_bound)


def aggregate_production_df(df, *, freq="M"):
    """Aggregate resolution of production data periods to target frequency `freq`.

    Examples
    --------
    >>> time = pd.period_range(start="2020-01-01", periods=100, freq="D")[::10]
    >>> production = [4.1, np.nan, 5.7, 6.8, 5.3, 2.9, 5. , 5.1, 4.2, 4.9]
    >>> time_on = [0.5, 0.9, 1. , 0.8, 0.8, np.nan, 0.9, 0, 0.6, 0.7]
    >>> df = pd.DataFrame({'well_id': [1] * 10, 'time': time,
    ...                    'production': production, 'time_on':time_on})
    >>> aggregate_production_df(df, freq="M")
       well_id     time  production   time_on
    0        1  2020-01        16.6  0.103226
    1        1  2020-02         8.2  0.027586
    2        1  2020-03        19.2  0.070968
    """

    assert set(df.columns) == {"well_id", "time", "production", "time_on"}

    assert freq == "M"  # Currently only aggregating from daily to monthly works
    assert df["time"].dtype.freq.freqstr == "D"
    assert (pd.isna(df["time_on"]) | (df["time_on"] >= 0) | (df["time_on"] <= 1)).all()

    df = (
        # Convert to datetime (needed for the Grouper to work, not sure why)
        df[["well_id", "time", "production", "time_on"]]
        .assign(time=lambda df: df["time"].dt.to_timestamp())
        .groupby(["well_id", pd.Grouper(key="time", freq="ME")])
        .sum()
        .reset_index()
        # Convert to monthly frequency and fix time_on
        .assign(
            # Note: The frequency "M" is called "MonthEnd", but "D" is "Day"
            time=lambda df: pd.PeriodIndex(df["time"], freq="M"),
            time_on=lambda df: df["time_on"] / df["time"].dt.daysinmonth,
        )
        .sort_values(["well_id", "time"])
    )
    return df


def robust_minimize(*, fun, x0, jac, args, **kwargs):
    """Run several optimization routines and return the best result.

    This function tries to deal with two errors can can occur, especially if
    the prior is very weak:
    1. Optimization can converge to a local minimum that is not the global
       minimum. To remedy this we run several algorithms and return the best
       result obtained.
    2. Optimization can return thetas that are very large or very small. This
       is typically to either the curve being increasing, in which case we are
       pushed toward the limit of what the parameters can be (very large or
       very small parameters). But these large/small parameters lead to over-
       and underflow in exp() when the curve is evaluated. Therefore some
       regularization helps. The visual difference is small.

    Examples
    --------
    >>> from scipy.optimize import rosen
    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    >>> res = robust_minimize(fun=rosen, x0=x0, jac=None, args=())
    >>> res.x
    array([0.99999931, 0.99999874, 0.99999753, 0.99999515, 0.99999028])
    """
    # This is hacky, but we prefer to be robust at the expense of some
    # computational time rather than returning wrong results.
    fun = copy.deepcopy(fun)
    jac = copy.deepcopy(jac)
    x0 = np.array(x0)

    bounds = [(-50, 50) for _ in x0]
    minf = functools.partial(sp.optimize.minimize, fun=fun, x0=x0, args=args)

    results = []
    with np.errstate(all="warn"):
        # BFGS is 5-10 times faster than the others, so we limit the iterations
        # on the other algorithms to make it faster.
        bfgs_result = minf(method="BFGS", jac=jac, **kwargs)
        results.append((bfgs_result.fun, "BFGS", bfgs_result))

        nm_result = minf(
            method="Nelder-Mead", bounds=bounds, options={"maxfev": 200}, **kwargs
        )
        results.append((nm_result.fun, "Nelder-Mead", nm_result))

        lbfgsb_result = minf(
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={"maxiter": 100},
            **kwargs,
        )
        results.append((lbfgsb_result.fun, "L-BFGS-B", lbfgsb_result))

    # Get best solution and return it
    _, algo, result = min(results)
    log.debug(f"Winning algo: {algo} {result.success} {result.x.round(3)}")

    # If the result is OOB, might be due to an increasing production rate
    # toward the end of the training data. This leads to bad estimates and
    # numerical errors. So we override the precision matrix and try again.
    if any(not (lb <= theta <= ub) for (theta, (lb, ub)) in zip(result.x, bounds)):
        log.warning(
            "Curve parameters out of bounds. Overriding prior for this fit. Consider increasing the prior strength."
        )
        log.debug(f"Theta too large (recursing): {result.x}")
        new_diag = np.diag(fun.precision_matrix) * 2.0 + 1e-6
        np.fill_diagonal(fun.precision_matrix, new_diag)
        return robust_minimize(fun=fun, x0=x0, jac=jac, args=args, **kwargs)

    return result


def clean_well_data(
    *,
    production: npt.NDArray[float],
    time: npt.NDArray[object],
    time_on: npt.NDArray[float] = None,
    return_log: bool = False,
):
    """Clean data. Returns a Bunch and optionally (Bunch, log).

    Examples
    --------

    Data that is bad and cannot be salvaged is removed:

    >>> production = np.array([10, -1, 0, np.nan, 1, 1, 1])
    >>> time = pd.period_range(start="2020-01-01", periods=7, freq="D")
    >>> time_on = np.array([1, 1, 1, 1, 0, 0, np.nan])
    >>> result = clean_well_data(production=production, time=time, time_on=time_on)
    >>> result.production
    array([10.])

    Some data can be saved by interpolation. Only the ones that are
    between two good data points. Here is [bad, good, bad, good]:

    >>> production = np.array([0, 10, 0, 5])
    >>> time_on = np.array([0, 0.5, 0, 0.5])
    >>> result = clean_well_data(production=production, time=time[:4], time_on=time_on)
    >>> for arrname, arr in result.items():
    ...     print(arrname, arr)
    production [10  5]
    time PeriodIndex(['2020-01-02', '2020-01-04'], dtype='period[D]')
    time_on [0.5 0.5]

    A full output log can be provided too.
    [kept, removed, kept, kept, interpolated, interpolated, kept, removed]
    np.array([10, np.nan, 8, 3,   6,       0,  4, 0])
    np.array([1,   0,     1, 0.5, np.nan, 0.5, 1, 0.5])

    >>> production = np.array([10, np.nan, 8, 3, 6, 0, 4, 0])
    >>> time = pd.period_range(start="2020-01-01", periods=8, freq="D")
    >>> time_on = np.array([1, 0, 1, 0.5, np.nan, 0.5, 1, 0.5])
    >>> result, log = clean_well_data(production=production, time=time,
    ...                               time_on=time_on, return_log=True)
    >>> for arrname, arr in result.items():
    ...     print(arrname, arr)
    production [10.          8.          3.          6.          2.33333333  4.        ]
    time PeriodIndex(['2020-01-01', '2020-01-03', '2020-01-04', '2020-01-05',
                 '2020-01-06', '2020-01-07'],
                dtype='period[D]')
    time_on [1.  1.  0.5 1.  0.5 1. ]
    >>> for reason_action_count in log:
    ...     print(reason_action_count)
    ('production > 0 AND time_on > 0', 'unchanged', 4)
    ('production  == NaN AND time_on == 0', 'removed', 1)
    ('production in (<=0, NaN) XOR time_on in (<=0, NaN)', 'interpolated', 2)
    ('production in (0, NaN) XOR time_on in (0, Nan)', 'removed', 1)
    """
    if isinstance(time, np.ndarray):
        assert np.allclose(np.sort(time), time), "Time must be sorted"
    else:
        assert isinstance(time.dtype, pd.PeriodDtype)
        assert (time.sort_values().values == time.values).all(), "Must be sorted"
    assert np.nanmin(time_on) >= 0
    assert np.nanmax(time_on) <= 1
    assert len(production) == len(time) == len(time_on)
    production = production.copy()
    time_on = time_on.copy()
    time = time.copy()

    # Here is a table of all cases. Some values are OK, some are
    # saved by interpolation, and bad data is thrown away.
    #                    production
    #                  NaN    <=0    >0
    #           NaN    bad    bad    intp
    # time_on   <=0    bad    bad    intp
    #            >0    inpt   inpt   OK
    #
    good_idx = (production > 0) & (time_on > 0)
    if np.any(good_idx):
        log = [("production > 0 AND time_on > 0", "unchanged", int(np.sum(good_idx)))]
    else:
        log = []

    # Data is bad if both production and time_on is bad
    # This can happen in four distinct ways, we report all of them here.
    functions = [lambda x: np.isclose(x, 0.0), lambda x: np.isnan(x)]
    functions = list(enumerate(functions))
    func_names = ["== 0", " == NaN"]

    for (i, fi), (j, fj) in itertools.product(functions, functions):
        bad_idx = fi(production) & fj(time_on)
        production = production[~bad_idx]
        time = time[~bad_idx]
        time_on = time_on[~bad_idx]
        if np.any(bad_idx):
            log.append(
                (
                    f"production {func_names[i]} AND time_on {func_names[j]}",
                    "removed",
                    int(np.sum(bad_idx)),
                )
            )

    # Return if all data was removed
    if not np.any((production > 0) & (time_on > 0)):
        result = Bunch(production=production, time=time, time_on=time_on)

        return (result, log) if return_log else result

    # Attempt to salvage more data by interpolation
    # Interpolation will only happen if the bad data point has one good
    # data point to the right of it and one good data point to the left.
    tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
    time_on_inpt, production_inpt = tsi.interpolate()

    def fixed(before, after):
        return (~(before > 0)) & (after > 0)

    fixed_idx = fixed(time_on, time_on_inpt) | fixed(production, production_inpt)
    if np.any(fixed_idx):
        log.append(
            (
                "production in (<=0, NaN) XOR time_on in (<=0, NaN)",
                "interpolated",
                int(np.sum(fixed_idx)),
            )
        )

    # Remove the data was that not salvaged by interpolation
    # Data is bad if either one of production or time_on is bad
    bad_idx = (~(time_on_inpt > 0)) | (~(production_inpt > 0))
    production = production_inpt[~bad_idx]
    time = time[~bad_idx]
    time_on = time_on_inpt[~bad_idx]
    if np.any(bad_idx):
        log.append(
            (
                "production in (0, NaN) XOR time_on in (0, Nan)",
                "removed",
                int(np.sum(bad_idx)),
            )
        )

    result = Bunch(
        production=production,
        time=time,
        time_on=time_on,
    )

    return (result, log) if return_log else result


def pairwise(iterable):
    """Return successive overlapping pairs taken from the input iterable.

    This function is not available from itertools in Python 3.9, only 3.10+.

    Examples
    --------
    >>> list(pairwise('ABCDE'))
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]
    """
    iterator = iter(iterable)
    a = next(iterator, None)

    for b in iterator:
        yield a, b
        a = b


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
