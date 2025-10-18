# %%
import pandas as pd

df = pd.read_csv("csv_output/DAILY.csv")
# %%
df.columns
# %%
df[df.ON_STREAM_HRS > 24].ON_STREAM_HRS = 24

# %%
use_cols = ["WELL_CODE", "DAYTIME", "ON_STREAM_HRS", "OIL_VOL"]
df = (
    df[use_cols]
    .assign(time_on=df["ON_STREAM_HRS"] / 24)
    .rename(
        columns={
            "WELL_CODE": "well_id",
            "DAYTIME": "time",
            "OIL_VOL": "production",
        }
    )
)

# %%
df.well_id.unique()

# %%
df.time = pd.to_datetime(df.time).dt.date
df.time_on.describe()
# %%
adca_cols = ["well_id", "time", "time_on", "production"]
df[adca_cols].to_csv("csv_output/DAILY_for_ADCA.csv", index=False)

# %%
