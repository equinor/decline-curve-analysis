import numpy as np
import pandas as pd

from dca.adca.dataloaders import filter_periods, validate
from dca.adca.well import Well
from dca.decline_curve_analysis import Arps


@validate
def generate_dummy_data(
    *,
    table: str,
    phases: list[str],
    frequency: str,
    well_ids: list[str],
    period_range: list[pd.Period],
    format: str,
) -> pd.DataFrame:
    """Returns dummy data, to be used for testing the code."""
    assert frequency == "monthly"

    rng = np.random.default_rng(42)

    dfs = []
    for well_id in well_ids:
        # Well data
        length = rng.integers(low=1, high=12 * 10)
        well_noise = rng.exponential(0.1)
        time = pd.period_range(end=pd.Period.now("M") - 1, periods=length, freq="M")

        # Remove some times randomly
        time = time[rng.random(len(time)) > 0.2]

        # Create curve (based on monthly oil data)
        theta = np.array([14.4, 3.6, -0.3]) + rng.standard_normal(3)

        # Create x and y
        x = np.arange(len(time), dtype=float) + 0.5
        mu = Arps(*theta).eval_log(x)
        y = np.exp(mu + rng.normal(loc=0, scale=well_noise, size=len(mu)))

        # Scale y by time_on
        time_on = rng.uniform(0, 1, size=len(time)) ** 0.2
        time_on[time_on < 0.66] = 1.0
        y = y * time_on

        # Create Well object and add its dataframe
        well = Well(id=well_id, production=y, time=time, time_on=time_on)
        dfs.append(well.to_df().drop(columns=["segment"]))

    return filter_periods(
        pd.concat(dfs, axis=0, ignore_index=True), period_range=period_range
    )
