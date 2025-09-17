import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dca.adca.well import Well, WellGroup
from dca.datasets import load_monthly_sodir_production

log = logging.getLogger("dca")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

# Load data to fit to
df = (
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
    .loc[
        lambda df: df.field.isin(
            [
                "GLITNE",
                "GULLFAKS SØR",
                "GYDA",
                "HEIDRUN",
                "NORNE",
                "STATFJORD",
            ]
        ),
        :,
    ]
)


# Create a group of wells
wells = WellGroup()
for field, df_field in df.groupby("field"):
    wells.append(
        Well(
            id=field,
            production=df_field.production.values,
            time_on=np.ones(len(df_field)),
            time=df_field.time,
        )
    )


# Tune hyperparameters
SPLIT = 0.8
optimal_parameters = wells.tune_hyperparameters(
    half_life=[12, 12 * 30], prior_strength=1, split=SPLIT, hyperparam_maxfun=10
)
print(f"{optimal_parameters=}")


# Fit using optimal hyperparameters
# Do note that we used the same test to determine hyperparameters,
# so we are overestimating how good the model would be in reality.
wells = wells.fit(**optimal_parameters, split=SPLIT)

# Create plots
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for ax, well in zip(axes.ravel(), wells):
    print(well.id, well.parameters_)
    well.plot(ax=ax, split=SPLIT, logscale=True, prediction=True)

plt.tight_layout()
plt.show()
