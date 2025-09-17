r"""

Time series preprocessing
-------------------------

Time series preprocessing is an important part of DCA. As such, we provide
some functionality for time series preparation and processing here.
There is no universally 'right' way to preprocess the time series,
it depends on what the purpose of the analysis/prediction is.

Please read the examples carefully and consider what is right for you.

Suppose we have data like this:

    >>> import numpy as np
    >>> import datetime as dt
    >>> from calendar import monthrange
    >>> dates = [dt.date(2020, i, 1) for i in range(1, 7)]
    >>> days_on = [20, 25, 15, 10, 15, 25]
    >>> time_on = [do / monthrange(d.year, d.month)[1] for
    ...           (do, d) in zip(days_on, dates)]
    >>> production = [100, 50, 25, 12, 6, 3]

Convert to Numpy:

    >>> dates = np.array(dates)
    >>> time_on = np.array(time_on)
    >>> production = np.array(production, dtype=float)
    >>> time_on.round(1)
    array([0.6, 0.9, 0.5, 0.3, 0.5, 0.8])

We now sketch three distinct ways of preprocessing the data.

Approach (1) to preprocessing: calendar time with no preprocessing
------------------------------------------------------------------

A simple approach. The disadvantage is that time_on is ignored.
The tuple (x, y, w) that goes into curve fitting is simply:

    >>> x = np.arange(len(time_on)) + 0.5
    >>> y = production
    >>> w = np.ones_like(production)
    >>> print(x, y, w, sep='\n')
    [0.5 1.5 2.5 3.5 4.5 5.5]
    [100.  50.  25.  12.   6.   3.]
    [1. 1. 1. 1. 1. 1.]

Approach (2) to preprocessing: production rates on producing time
-----------------------------------------------------------------

This approach puts  data on producing days, so if we desire to predict the
next calendar year this approach is not appropriate.

    >>> x, y, w = to_producing_time(time_on, production)
    >>> print(x.round(1), y.round(1), w.round(1), sep='\n')
    [0.3 1.1 1.7 2.2 2.6 3.2]
    [155.   58.   51.7  36.   12.4   3.6]
    [0.6 0.9 0.5 0.3 0.5 0.8]

If `time_on` is 1.0, then this approaches reduces to the previous one:

    >>> x, y, w = to_producing_time(np.ones_like(time_on), production)
    >>> print(x.round(1), y.round(1), w.round(1), sep='\n')
    [0.5 1.5 2.5 3.5 4.5 5.5]
    [100.  50.  25.  12.   6.   3.]
    [1. 1. 1. 1. 1. 1.]


"""

import functools
import logging
import operator

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def mask_leading_zeros(production: npt.NDArray[float]) -> npt.NDArray[bool]:
    """Return a mask indicating which values to keep in order to remove leading
    zeros.

    Examples
    --------
    >>> production = np.array([0, 0, 1, 2, 0, 1, 0, 0])
    >>> mask = mask_leading_zeros(production)
    >>> production[mask]
    array([1, 2, 0, 1, 0, 0])
    """
    assert isinstance(production, np.ndarray)
    assert production.ndim == 1

    return np.arange(len(production)) > np.argmax(production > 0) - 1


def to_producing_time(
    time_on: npt.NDArray[float],
    production: npt.NDArray[float],
    center_intervals: bool = True,
    return_mask: bool = False,
) -> tuple:
    """Returns (x, y, w) for curve fitting.

    Transform production per time period to production rate. This function
    also creates an x-axis as well as data-point weights for curve fitting.

    Returns a tuple with arrays (cumulative_time_on, production_rate, time_on),
    which may be used for curve fitting as (x, y, weights).

    Periods with no time on are removed.

    Examples
    --------
    Assume a well produces 10 units per time period if on all the time. If it's
    not on all the time, we can adjust the time series to account for this:

        >>> time_on = np.array([1, 0, 0.5, 0.2])
        >>> production = np.array([10, 0, 5, 2])
        >>> x, y, w = to_producing_time(time_on, production, center_intervals=False)
        >>> x
        array([1. , 1.5, 1.7])
        >>> y
        array([10., 10., 10.])

    Intervals may be centered as preparation for curve fitting.

        >>> x, y, w = to_producing_time(time_on, production, center_intervals=True)
        >>> x
        array([0.5 , 1.25, 1.6 ])

    ValueErrors are raised if the data is inconsistent:

        >>> time_on = np.array([1, 0])
        >>> production = np.array([1, 2])
        >>> to_producing_time(time_on, production)
        Traceback (most recent call last):
          ...
        ValueError: Found production > 0, when time_on == 0.

    However, the opposite is OK:

        >>> time_on = np.array([1, 1])
        >>> production = np.array([1, 0])
        >>> x, y, z = to_producing_time(time_on, production)
    """

    # Validate input arguments
    production = np.array(production)
    time_on = np.array(time_on)
    assert isinstance(center_intervals, bool)

    assert np.all(np.isfinite(production)), "Values must be finite and not NaN"
    assert np.all(np.isfinite(time_on)), "Values must be finite and not NaN"

    # Equal shapes and vectors
    assert production.shape == time_on.shape
    assert production.ndim == 1

    # The time on should be a fraction
    assert np.min(time_on) >= 0, "Time on must be in range [0, 1]"
    assert np.max(time_on) <= 1, "Time on must be in range [0, 1]"

    assert np.min(production) >= 0, "Production must be non-negative"

    # When production is zero, time_on should be zero
    if not np.all((production == 0) >= (time_on == 0)):
        raise ValueError("Found production > 0, when time_on == 0.")

    # Filter out zero production periods. This must be done to ensure that
    # we do not divide by zero when computing the production rate below.
    mask = time_on > 0
    production = production[mask]
    time_on = time_on[mask]

    # The production rate is the production per time period divided by the time on
    production_rate = production / time_on
    cumulative_time_on = np.cumsum(time_on)

    # Center each interval. This is done since we want the integral from 0 to n
    # to equal the sum of the n first data points. Consider a single point.
    # The integral from 0 to 1 is not equal to the data point at 1, since the
    # decline curves decreases. If we shift the data point to 0.5 the integral
    # from 0 to 1 will more closely match the data value.
    if center_intervals:
        cumulative_time_on = cumulative_time_on - time_on / 2

    if return_mask:
        return cumulative_time_on, production_rate, time_on, mask
    else:
        return cumulative_time_on, production_rate, time_on


class TimeSeriesInterpolator:
    """This class imputes values in `time_on` and `production` that are either
    (1) non-positive or (2) NaN. It does so by assuming that the production rate,
    which is defined as `production` / `time_on`, is a piecewise linear function.


    Parameters
    ----------
    time_on : npt.NDArray[float]
        An array with time on per period, given as a fraction between 0 and 1.
        Values that are non-positive or NaN will be interpolated if there
        exists a production value for the same index, and there exists one
        index before and after where both time_on and production is known.
        This is needed to infer the production rate.
    production : npt.NDArray[float]
        An array with production per period. Values that are non-positive or
        NaN will be interpolated, using the same rules as for `time_on`.
    logspace : bool, optional
        Whether or not to interpolate the production rate in logspace or not.
        If True, then production rates [1, NaN, 100] will be interpolated to 10
        in the missing index. If False it will be interpolated to 50.5.
        The default is False.

    Examples
    --------

    A simple example with missing production and constant production rate:

    >>> time_on = np.array([0.4, 0.6, 0.8])
    >>> production = np.array([40, np.nan, 80])
    >>> tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
    >>> tsi.interpolate()
    (array([0.4, 0.6, 0.8]), array([40., 60., 80.]))

    The situation becomes more complex when the production rate is not constant
    and time_on is missing. When that is the case, the interpolation grid is
    itself unknown. Consider this example:

    >>> time_on = np.array([0.4, np.nan, 0.8])
    >>> production = np.array([40, 90, 160])
    >>> tsi = TimeSeriesInterpolator(time_on=time_on, production=production)

    A first thought might be to interpolate using a grid where NaN equals 1.0:

    >>> x = np.nan_to_num(time_on, nan=1.0)
    >>> x = np.nancumsum(x) - x / 2
    >>> x
    array([0.2, 0.9, 1.8])
    >>> tsi.fixed_grid(x)
    (array([0.4       , 0.62608696, 0.8       ]), array([ 40,  90, 160]))

    But now we have an estimate for `time_on`, and that estimate can be used to
    interpolate on a grid that we know is more correct than what we just used.
    If we iterate on the interpolation, we will converge to:

    >>> tsi.interpolate()
    (array([0.4       , 0.63425855, 0.8       ]), array([ 40,  90, 160]))
    """

    def __init__(
        self,
        *,
        time_on: npt.NDArray[float],
        production: npt.NDArray[float],
        logspace=False,
    ):
        # Convert arguments to arrays, do not mutate inputs
        time_on, production = np.array(time_on), np.array(production)
        assert isinstance(logspace, bool)

        # Equal shapes and vectors
        assert time_on.shape == production.shape
        assert production.ndim == 1

        # The time on should be a fraction
        assert np.nanmin(time_on) >= 0, "Time on must be in range [0, 1]"
        assert np.nanmax(time_on) <= 1, "Time on must be in range [0, 1]"

        self.time_on = time_on
        self.production = production
        if logspace:
            self.func = np.log
            self.inv_func = np.exp
        else:
            self.func = lambda x: x
            self.inv_func = lambda x: x

        # Values that we want to fix, conditioned on the other data
        # One idea was to bootstrap the estimates, and use previously imputed
        # values to impute new values. This did not help however, and it seems
        # more sensible to base estimates on interpolating known values only.
        self.idx_fix_time_on = (~(self.time_on > 0)) & (self.production > 0)
        self.idx_fix_production = (~(self.production > 0)) & (self.time_on > 0)
        self.idx_both = (self.production > 0) & (self.time_on > 0)

    def interpolate_time_on(self, time_on, production, x):
        """Return `time_on` with interpolated values, using interpolation grid x."""
        assert len(time_on) == len(production) == len(x)
        time_on = time_on.copy()
        fix = self.idx_fix_time_on

        if not np.any(self.idx_both):
            return time_on

        # Interpolate rates - only use good information
        m = self.idx_both
        rates = self.func(production[m] / time_on[m])
        rates = self.inv_func(np.interp(x, x[m], rates, left=np.nan, right=np.nan))

        time_on[fix] = np.minimum(1, production[fix] / rates[fix])
        return time_on

    def interpolate_production(self, time_on, production, x):
        """Return `production` with interpolated values, using interpolation grid x."""
        assert len(time_on) == len(production) == len(x)
        production = production.copy()
        fix = self.idx_fix_production

        if not np.any(self.idx_both):
            return production

        # Interpolate rates - only use good information
        m = self.idx_both
        rates = self.func(production[m] / time_on[m])
        rates = self.inv_func(np.interp(x, x[m], rates, left=np.nan, right=np.nan))

        production[fix] = rates[fix] * time_on[fix]
        return production

    def interpolate(self, maxiter=99):
        """Interpolate by iterating back and forth until convergence.

        Examples
        --------
        The production rate is given by f(x) = 10 - 2 * x

        >>> time_on = np.array([0.4, 0.6, 0.8, 1.0])
        >>> x = np.cumsum(time_on) - time_on/2
        >>> production_rate = 10  - 2 * x
        >>> production = production_rate * time_on
        >>> production
        array([3.84, 5.16, 5.76, 5.4 ])

        Fill with some missing values:

        >>> production[1] = np.nan
        >>> time_on[2] = np.nan

        Interpolate it:

        >>> tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        >>> time_on, production = tsi.interpolate()
        >>> time_on
        array([0.4, 0.6, 0.8, 1. ])
        >>> production
        array([3.84, 5.16, 5.76, 5.4 ])
        """

        if not (np.any(self.idx_fix_time_on) or np.any(self.idx_fix_production)):
            return self.time_on.copy(), self.production.copy()

        if not np.any(self.idx_both):
            return self.time_on.copy(), self.production.copy()

        # Initial values for the iterative algorithm
        time_on_prev = self.time_on.copy()
        production_prev = self.production.copy()

        for _ in range(1, maxiter + 1):
            # set up an interpolation grid based on time_on
            x = np.nan_to_num(time_on_prev, copy=True, nan=np.nanmean(time_on_prev))
            x = np.cumsum(x) - x / 2

            # We can choose to condition on 'time_on' or 'production' first,
            # but in practice it does not matter. The number of iterations
            # that are needed is the same either way
            production = self.interpolate_production(time_on_prev, production_prev, x=x)
            time_on = self.interpolate_time_on(time_on_prev, production, x=x)

            # Convergence checks
            close_time = np.allclose(time_on, time_on_prev, equal_nan=True, rtol=1e-8)
            close_prod = np.allclose(
                production, production_prev, equal_nan=True, rtol=1e-8
            )

            if close_time and close_prod:
                break

            time_on_prev, production_prev = time_on, production
        else:
            log.warning(
                f"Time series interpolation did not converge in {maxiter} iters."
            )

        return time_on, production

    def fixed_grid(self, x):
        """Solve the interpolation with using a static interpolation grid.

        Examples
        --------
        The true production rate is [24, 20, 16, 12], but we observe:

        >>> time_on = np.array([1, np.nan, 0.6, 0.4]) # True: [1, 0.8, 0.6, 0.4]
        >>> production = np.array([24, 16, np.nan, 4.8]) # True: [24, 16, 9.6, 4.8]
        >>> tsi = TimeSeriesInterpolator(time_on=time_on, production=production)
        >>> x = np.arange(4) # Naive equidistant grid
        >>> time_on, production = tsi.fixed_grid(x)
        >>> time_on
        array([1. , 0.8, 0.6, 0.4])
        >>> production
        array([24. , 16. ,  9.6,  4.8])
        >>> production / time_on
        array([24., 20., 16., 12.])

        With a better estimated grid:

        >>> x = np.array([0.5, 1.5, 2.3, 2.8]) # Assumes missing time_on = 1
        >>> time_on, production = tsi.fixed_grid(x)
        >>> time_on
        array([1.        , 0.85185185, 0.6       , 0.4       ])
        >>> x = np.array([0.5, 1.4, 2.1, 2.6]) # Assumes missing time_on = 0.8
        >>> time_on, production = tsi.fixed_grid(x)
        >>> time_on
        array([1.        , 0.84848485, 0.6       , 0.4       ])
        """
        x = np.array(x, dtype=float)
        assert len(x) == len(self.time_on) == len(self.production)

        if not (np.any(self.idx_fix_time_on) or np.any(self.idx_fix_production)):
            return self.time_on.copy(), self.production.copy()

        if not np.any(self.idx_both):
            return self.time_on.copy(), self.production.copy()

        time_on = self.interpolate_time_on(self.time_on, self.production, x=x)
        production = self.interpolate_production(time_on, self.production, x=x)
        return time_on, production


def preprocess_timeseries(
    time_on: npt.NDArray[float],
    production: npt.NDArray[float],
    return_errors: bool = False,
) -> tuple:
    """Preprocess a time series with `time_on` and `production`, dealing with
    zeros, NaNs and other inconsistencies.

    This function does cleanup in preparation of DCA. For instance, if `production`
    was positive but `time_on` is NaN, then we assume `time_on` is actually
    1. Another example: if `time_on` is positive, but `production`
    is zero, then we assume `time_on` is actually zero (since there were no
    producing days).

    The function returns both processed vectors, as well as a mask indicating
    which values to keep. This is because each period might be associated with
    a date, and processing the inputs might involve using the mask to e.g. mask
    dates as well. So the function returns arrays and a mask, expecting the user
    to mask the arrays after they are returned - and use the mask for other
    purposes if required.

    Parameters
    ----------
    time_on : np.ndarray
        Fraction of each time period that the well was on, between 0 and 1.
    production : np.ndarray
        Production per time period.
    return_errors: bool
        Whether or not to return a list of strings with error summary.

    Returns
    -------
    time_on : np.ndarray
        A new array with no NaNs and "imputed" values.
    production : np.ndarray
        A new array with no NaNs and "imputed" values.
    mask : np.ndarray with boolean entries
        A mask indicating with values in the arrays to keep.

    Examples
    --------

    An introductory example:

    >>> time_on = np.array([1, 1, 0])
    >>> production = np.array([50, 0, 50])
    >>> preprocess_timeseries(time_on,  production)
    (array([1, 1, 1]), array([50,  0, 50]), array([ True, False,  True]))

    A full example covering all 3 x 3 = 9 cases:

    >>> n = np.nan
    >>> time = np.array([0.5, 0, 0, 0, 1, 1, 1, n, n, n, 1])
    >>> prod = np.array([0,   1, 0, n, 1, 0, n, 1, 0, n, -1])
    >>> time_on, production, mask = preprocess_timeseries(time, prod)
    >>> time_on # Imputed values
    array([0.5, 1. , 0. , 0. , 1. , 1. , 1. , 1. , 0. , 0. , 0. ])
    >>> production # Imputed values
    array([0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0.])
    >>> mask * 1.0 # Values to keep
    array([0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0.])

    With explanations:

    >>> *_, errors = preprocess_timeseries(time, prod, return_errors=True)
    >>> for (t_i, p_i, e_i) in zip(time, prod, errors):
    ...     print(f'Time: {t_i} Prod: {p_i} Error: {e_i}')
    Time: 0.5 Prod: 0.0 Error: ((time_on > 0) & (production <= 0))
    Time: 0.0 Prod: 1.0 Error: (time_on <= 0) & (production > 0)
    Time: 0.0 Prod: 0.0 Error: ((time_on <= 0) & (production <= 0))
    Time: 0.0 Prod: nan Error: (time_on <= 0) & isnan(production)
    Time: 1.0 Prod: 1.0 Error: None
    Time: 1.0 Prod: 0.0 Error: ((time_on > 0) & (production <= 0))
    Time: 1.0 Prod: nan Error: (time_on > 0) & isnan(production)
    Time: nan Prod: 1.0 Error: isnan(time_on) & (production > 0)
    Time: nan Prod: 0.0 Error: isnan(time_on) & (production <= 0)
    Time: nan Prod: nan Error: isnan(time_on) & isnan(production)
    Time: 1.0 Prod: -1.0 Error: production < 0

    Before and after:

    >>> result = preprocess_timeseries(time, prod, return_errors=True)
    >>> time_on, production, mask, errors = result
    >>> time_on[~mask] = 0 # Soft delete by setting time_on to zero
    >>> before_after = zip(time, prod, time_on, production)
    >>> for (t_b, p_b, t_a, p_a) in before_after:
    ...     print(f'BEFORE: Time: {t_b} Prod: {p_b} -> AFTER: Time: {t_a} Prod: {p_a}')
    BEFORE: Time: 0.5 Prod: 0.0 -> AFTER: Time: 0.0 Prod: 0.0
    BEFORE: Time: 0.0 Prod: 1.0 -> AFTER: Time: 1.0 Prod: 1.0
    BEFORE: Time: 0.0 Prod: 0.0 -> AFTER: Time: 0.0 Prod: 0.0
    BEFORE: Time: 0.0 Prod: nan -> AFTER: Time: 0.0 Prod: 0.0
    BEFORE: Time: 1.0 Prod: 1.0 -> AFTER: Time: 1.0 Prod: 1.0
    BEFORE: Time: 1.0 Prod: 0.0 -> AFTER: Time: 0.0 Prod: 0.0
    BEFORE: Time: 1.0 Prod: nan -> AFTER: Time: 0.0 Prod: 0.0
    BEFORE: Time: nan Prod: 1.0 -> AFTER: Time: 1.0 Prod: 1.0
    BEFORE: Time: nan Prod: 0.0 -> AFTER: Time: 0.0 Prod: 0.0
    BEFORE: Time: nan Prod: nan -> AFTER: Time: 0.0 Prod: 0.0
    BEFORE: Time: 1.0 Prod: -1.0 -> AFTER: Time: 0.0 Prod: 0.0


    """
    # Convert arguments to arrays, do not mutate inputs
    time_on, production = np.array(time_on), np.array(production)

    # Equal shapes and vectors
    assert time_on.shape == production.shape
    assert production.ndim == 1

    # The time on should be a fraction
    assert np.nanmin(time_on) >= 0, "Time on must be in range [0, 1]"
    assert np.nanmax(time_on) <= 1, "Time on must be in range [0, 1]"

    # Log the errors into this array:
    NO_ERROR = "None"
    error_text = np.array([NO_ERROR] * len(time_on), dtype=object)

    # A special case for negative production values
    mask = production < 0
    time_on[mask] = 0
    production[mask] = 0
    error_text[mask] = "production < 0"

    # There are 9 cases, since there are 3 inputs
    # (production, time_on and days_in_month)
    # For each of there 3 inputs, there are three options: Nan, <= 0 and > 0
    # Here we systematically deal with each case in turn:

    # (1) Cases when time_on is NaN
    # -----------------------------

    # (1a) If both are NaN, set to zero (later on they will be removed by mask)
    mask_both_NaN = np.isnan(time_on) & np.isnan(production)
    mask_not_both_NaN = ~mask_both_NaN
    time_on[mask_both_NaN] = 0
    production[mask_both_NaN] = 0

    mask = mask_both_NaN & (error_text == NO_ERROR)
    error_text[mask] = "isnan(time_on) & isnan(production)"

    # (1b) If time_on is NaN and production is <= 0, assume zeros days on
    mask = np.isnan(time_on) & (production <= 0)
    time_on[mask] = 0

    mask = mask & (error_text == NO_ERROR)
    error_text[mask] = "isnan(time_on) & (production <= 0)"

    # (1c) If time_on is NaN and production is > 0, then set time_on to 1 (best guess)
    mask = np.isnan(time_on) & (production > 0)
    time_on[mask] = 1.0

    mask = mask & (error_text == NO_ERROR)
    error_text[mask] = "isnan(time_on) & (production > 0)"

    # (2) Cases when time_on <= 0
    # ---------------------------

    # (2a) If time_on is <= 0 and production is NaN, then set production to 0
    mask = (time_on <= 0) & np.isnan(production)
    production[mask] = 0

    mask = mask & (error_text == NO_ERROR)
    error_text[mask] = "(time_on <= 0) & isnan(production)"

    # (2b) If time_on is <= 0 and production is <= 0, then remove data point
    mask_both_not_nonpositive = ~((time_on <= 0) & (production <= 0))

    mask = (~mask_both_not_nonpositive) & (error_text == NO_ERROR)
    error_text[mask] = "((time_on <= 0) & (production <= 0))"

    # (2c) If time_on is <= 0 and production is > 0,
    # then set time_on to 1.0 (best guess)
    mask = (time_on <= 0) & (production > 0)
    time_on[mask] = 1.0

    mask = mask & (error_text == NO_ERROR)
    error_text[mask] = "(time_on <= 0) & (production > 0)"

    # (3) Cases when time_on > 0
    # --------------------------

    # (3a) If time_on is > 0 and production is NaN, then set production to 0
    mask = (time_on > 0) & np.isnan(production)
    production[mask] = 0

    mask = mask & (error_text == NO_ERROR)
    error_text[mask] = "(time_on > 0) & isnan(production)"

    # (3b) If time_on is > 0 and production is <= 0, then then data point will
    # be removed

    # TODO: Remove this, or keep it as it could indicate zero production?
    mask_positive_time_on_but_no_production = ~((time_on > 0) & (production <= 0))
    mask = ~mask_positive_time_on_but_no_production

    mask = mask & (error_text == NO_ERROR)
    error_text[mask] = "((time_on > 0) & (production <= 0))"

    # (3c) If time_on is > 0 and production is > 0, then do nothing

    # Other cases
    # -----------

    # From the start of the data, remove the first periods with zero productions
    mask_not_zero_at_start = np.arange(len(production)) > np.argmax(production > 0) - 1

    mask = (~mask_not_zero_at_start) & (error_text == NO_ERROR)
    error_text[mask] = "leading 0 production"

    # Combine all masks to a common mask of what to keep
    total_mask = functools.reduce(
        operator.and_,
        [
            mask_not_both_NaN,
            mask_not_zero_at_start,
            mask_both_not_nonpositive,
            mask_positive_time_on_but_no_production,
        ],
    )

    assert np.all(np.isfinite(production)), "Production must be finite and not NaN"
    assert np.all(np.isfinite(time_on)), "Time on must be finite and not NaN"

    if return_errors:
        return time_on, production, total_mask, error_text
    else:
        return time_on, production, total_mask


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
