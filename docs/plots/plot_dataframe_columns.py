import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dca.adca.well import Well
from dca.datasets import load_monthly_sodir_production

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LOGSCALE = False

# Load data to fit to
df_production = (
    load_monthly_sodir_production()
    .groupby(["prfInformationCarrier", "prfYear", "prfMonth"])["prfPrdOilNetMillSm3"]
    .sum()
    .rename("production")
    .reset_index()
    .rename(
        columns={
            "prfInformationCarrier": "field",
            "prfYear": "year",
            "prfMonth": "month",
        }
    )
    .assign(day=1)
    .assign(
        time=lambda df: pd.to_datetime(df[["year", "month", "day"]]).dt.to_period(
            freq="M"
        )
    )
    .loc[lambda df: df.production > 1e-4]
    .loc[lambda df: df.field == "STATFJORD", :]
    .sort_values("time")
)

# There is no train/test split on the output. Remove some recent data
df_production = df_production.iloc[: int(len(df_production) * 0.5), :]


# Create a well
well = Well(
    id=df_production.field.values[0],
    production=df_production.production.values,
    time_on=np.ones(len(df_production)),
    time=df_production.time,
)


# Fit the well
well = well.fit(half_life=36, prior_strength=1e-3, split=1.0)


# Get dataframe with forecasts
df = well.to_df(forecast_periods=len(df_production), q=[0.1, 0.5, 0.9], simulations=999)


df["time"] = df["time"].dt.to_timestamp()


for well_id, df_i in df.groupby("well_id"):
    df_i = df_i.sort_values("time")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(figsize=(7, 6), nrows=2, ncols=1, sharex=True)

    # ================================================
    title = r"""Production rate columns: evaluations of the curve 
function $f(t_i, \theta)$ at every period. The uncertainty is 
elementwise and should NOT be summed."""
    ax1.set_title(title)

    df_i_history = df_i.dropna(subset=["production"])
    ax1.plot(
        df_i_history.time, df_i_history.production, color=COLORS[0], label="production"
    )

    df_i_future = df_i.loc[lambda df: df.production.isnull(), :]
    ax1.plot(
        df_i_future.time,
        df_i_future.forecasted_production,
        color=COLORS[1],
        label="forecasted_production",
    )

    for col in df_i.columns:
        if "forecasted_production_P" in col:
            ax1.plot(
                df_i_future.time,
                df_i_future[col].values,
                color=COLORS[1],
                ls="--",
                label=col,
            )

    ax1.grid(True, ls="--", alpha=0.33)
    ax1.set_yscale("log" if LOGSCALE else "linear")
    ax1.legend()

    # ================================================
    title = r"""Cumulative production columns: properly summed
production uncertainty using Monte Carlo simulations."""
    ax2.set_title(title)

    ax2.plot(
        df_i_history.time,
        df_i_history.production.cumsum(),
        color=COLORS[0],
        label="cumsum(production)",
    )

    ax2.plot(
        df_i_future.time,
        df_i_future.cumulative_production,
        color=COLORS[1],
        label="cumulative_production",
    )

    for col in df_i.columns:
        if "cumulative_production_P" in col:
            ax2.plot(
                df_i_future.time,
                df_i_future[col].values,
                color=COLORS[1],
                ls="--",
                label=col,
            )

    ax2.grid(True, ls="--", alpha=0.33)
    ax2.set_yscale("log" if LOGSCALE else "linear")
    ax2.legend()

    fig.tight_layout()
    plt.show()
