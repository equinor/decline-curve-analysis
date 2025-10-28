"""
Dataloaders are responsible for fetching data from a source system and returning
a pandas DataFrame with a specified format.

Columns
-------
 - well_id : ID as it appears in the source system
 - time : pd.Period of each period, signifies total production within that period
          Monthly: '01.01.2021', '01.02.2021', '01.03.2021'
          Daily: '01.01.2021', '02.01.2021', '03.01.2021'
 - production : total production within each period (units do not matter)
 - time_on : fraction of each period that the well produced

"""

import collections
import functools
import logging

import numpy as np
import pandas as pd

from dca.adca.utils import aggregate_production_df
from dca.datasets import load_monthly_sodir_production
from dca.timeseries import TimeSeriesInterpolator

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def report_data_quality(df, per_well=False):
    """Return a string with data quality issues in Markdown format.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'well_id': ['A', 'A', 'A', 'B', 'B'],
    ...                    'time': [pd.Period('2000-07-29', 'D'),
    ...                             pd.Period('2011-05-10', 'D'),
    ...                             pd.Period('1998-10-28', 'D'),
    ...                             pd.Period('2007-01-01', 'D'),
    ...                             pd.Period('2016-08-19', 'D')],
    ...                    'production': [1646.0, 0.0, None, 216.0, 573.0],
    ...                    'time_on': [1.0, 0.0, 1.0, 1.0, None]})
    >>> report_data_quality(df)
      production time_on  count  total  percentage
    1         >0     NaN      1      5        20.0
    2        NaN      >0      1      5        20.0
    3        ~=0     ~=0      1      5        20.0
    0         >0      >0      2      5        40.0
    """
    df = df.copy().sort_values("well_id")
    groupby = (["well_id"] if per_well else []) + ["production", "time_on"]

    def categorize(series):
        # Function to categorize the values
        array = series.copy().to_numpy()
        output = np.array(["UNKNOWN"] * len(array))
        output[np.isnan(array)] = "NaN"
        zero_mask = np.isclose(array, 0.0)
        output[zero_mask] = "~=0"
        output[(array > 0) & (~zero_mask)] = ">0"
        output[(array < 0) & (~zero_mask)] = "<0"
        return output

    # Apply the function to both columns
    df["production"] = categorize(df["production"])
    df["time_on"] = categorize(df["time_on"])

    # Calculate the combination counts
    sort_key = (["well_id"] if per_well else []) + ["count"]
    df_combs = (
        df.groupby(groupby, sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(sort_key)
    )

    if per_well:
        # Join in total rows per well
        df_combs = df_combs.merge(
            df["well_id"].value_counts().rename("total").to_frame(),
            how="inner",
            on=["well_id"],
        )
    else:
        df_combs = df_combs.assign(total=len(df))

    # Calculate the percentages
    df_combs["percentage"] = ((df_combs["count"] / df_combs["total"]) * 100).round(3)

    return df_combs


def filter_periods(df, *, period_range):
    """Filter a loaded DataFrame by period range [start, stop)."""

    # Split out the dates
    start, stop = period_range
    if not pd.isna(start):
        df = df[df.time >= pd.Period(start)]
    if not pd.isna(stop):
        df = df[df.time < pd.Period(stop)]

    return df


def validate(dataloader_func):
    """Amends a dataloader function with input/output checks."""

    @functools.wraps(dataloader_func)
    def validated_dataloader_func(
        *,
        table: str,
        phases: list[str],
        frequency: str,
        well_ids: list[str],
        period_range: list,
        format: str,
    ):
        # Input argument validation
        assert set(phases).issubset({"gas", "oil", "cond", "wat"})
        assert frequency in ("daily", "monthly")
        assert all(isinstance(well_id, str) for well_id in well_ids)
        assert isinstance(period_range, list)
        assert len(period_range) == 2
        assert all(
            (isinstance(period, pd.Period) or pd.isna(period))
            for period in period_range
        )
        assert (format is None) or isinstance(format, str)

        # Check that there are no duplicate well IDs
        well_id_counts = collections.Counter(well_ids)
        duplicate_well_ids = [k for (k, v) in well_id_counts.items() if v > 1]
        if duplicate_well_ids:
            raise ValueError(
                f"Duplicate `well_id` not allowed: {set(duplicate_well_ids)}"
            )

        # Run the original function
        df = dataloader_func(
            table=table,
            phases=phases,
            frequency=frequency,
            well_ids=well_ids,
            period_range=period_range,
            format=format,
        )

        # Output argument (dataframe) validation

        # Verify column names
        assert set(df.columns) == {"well_id", "time", "production", "time_on"}

        # Verify column data (here NaNs are ignored)
        assert df["time_on"].max() <= 1.0
        assert df["time_on"].min() >= 0.0
        assert df["production"].min() >= 0.0
        assert isinstance(df["time"].dtype, pd.PeriodDtype)
        assert pd.api.types.is_numeric_dtype(df["production"].dtype)
        assert not df["well_id"].str.contains(",").any(), (
            "The character ',' cannot be used in well_id"
        )

        assert not df.duplicated(subset=["well_id", "time"]).any()

        # D = Day, ME = MonthEnd
        assert df["time"].dtype.freq.freqstr in ("D", "ME")

        # Verify that config and data has the same wells
        not_retrieved = set(well_ids) - set(df.well_id.unique())
        assert not_retrieved == set(), f"Wells not found in table: {not_retrieved}"

        extra_wells = set(df.well_id.unique()) - set(well_ids)
        if dataloader_func.__name__ != "load_file":
            assert extra_wells == set(), (
                f"Extra wells fetched, but not asked for: {extra_wells}"
            )
            assert set(df.well_id.unique()) == set(well_ids)
        else:
            if extra_wells:
                log.info(f"Extra wells (in file, but not used): {extra_wells}")
            df = df[df.well_id.isin(well_ids)]  # Filter returned data

        # If the user asked for monthly information, then we do not want to use
        # the current month, since it is not over and production values will
        # look artificially low
        freq = {"monthly": "M", "daily": "D"}[frequency]
        assert not (df.time >= pd.Period.now(freq)).any()

        return df

    return validated_dataloader_func


@validate
def load_file(
    *,
    table: str,
    phases: list[str],
    frequency: str,
    well_ids: list[str],
    period_range: list[pd.Period],
    format: str,
) -> pd.DataFrame:
    """
    Read data from a file and return DataFrame.

    The input file must have either daily or monthly data. The `format` argument
    is passed to pandas.to_datetime, and is typically "%Y-%m-%d" or similar.
    The `period_range` must match the `frequency`.
    """
    freq = {"monthly": "M", "daily": "D"}[frequency]

    # Data must have year and month, and could possibly have day
    assert (format is None) or all(substr in format for substr in ("%m", "%Y"))

    assert isinstance(table, str)
    assert table.endswith(".csv")
    df = pd.read_csv(table)[["well_id", "time", "production", "time_on"]]
    assert not df.empty

    # Convert dates to periods
    log_idx = pd.Series(range(len(df))).sample(3, random_state=0)
    log.info("Printing some random times. Please verify the parsing.")
    log.info(f"Time before parsing: {df.iloc[log_idx, :].loc[:, 'time'].tolist()}")
    df = df.assign(time=pd.to_datetime(df.time, format=format, errors="raise"))

    # Infer the period frequency to convert to before any poential aggregation.
    # This is a bit hacky, and assumes nice user inputs.
    input_freq = "M" if (df.time.dt.day == 1).all() else "D"
    df = df.assign(time=lambda df: pd.PeriodIndex(df.time, freq=input_freq))
    log.info(f"Time after parsing:  {df.iloc[log_idx, :].loc[:, 'time'].tolist()}")

    # Sum to monthly if the data is not already on monthly resolution
    if (frequency == "monthly") and df["time"].dtype.freq.freqstr != "ME":
        log.info("Aggregating to monthly.")
        df = aggregate_production_df(df, freq="M")

    # Only keep data from before this current month/day
    df = df.loc[lambda df: df.time < pd.Period.now(freq), :]

    return filter_periods(
        df[["well_id", "time", "production", "time_on"]], period_range=period_range
    )


@validate
def load_PDM_data(
    *,
    table: str,
    phases: list[str],
    frequency: str,
    well_ids: list[str],
    period_range: list[pd.Period],
    format: str,
) -> pd.DataFrame:
    """Read data from PDM (production data mart) and return DataFrame.

    Data quality
    ------------

    There are data errors in WB_PROD_DAY, for instance, for one well we count:

          oil     time_on  count  percentage
    0       0           0   3128       37.07  <- OK
    1       0          >0     29        0.34
    2       0         NaN      9        0.10
    3      >0           0   1743       20.66
    4      >0          >0   3527       41.80  <- OK
    5     NaN         NaN      1        0.01

    By far the most common data error is time_on==0 when the well is producing.
    The correct way to deal with this is to treat time_on as NULL / MISSING.
    """
    assert isinstance(well_ids, list)
    assert table in ("PDMVW.WB_PROD_DAY",)

    where = f"IN {tuple(well_ids)}" if len(well_ids) > 1 else f"= {well_ids[0]!r}"

    sql = f"""
        SELECT
            wb_uwbi,
            prod_day,
            wb_oil_vol_sm3 as oil,
            wb_cond_vol_sm3 as cond,
            wb_gas_vol_sm3 as gas,
            wb_water_vol_m3 as wat,
            on_stream_hrs as time_on
        FROM {table}
        WHERE wb_uwbi {where}
    """
    from pdm_datareader import tools  # Only import as needed

    log.info("Reading Production Data Mart (PDM).")
    df = tools.query(sql)
    if df.empty:
        msg = "Connection to PDM was successful, but zero rows were returned from the database.\n"
        msg += "Check that well IDs in the .yaml config file match column WB_UWBI in PDM."
        raise Exception(msg)

    df = df.assign(
        well_id=lambda df: df.wb_uwbi,
        production=lambda df: pd.to_numeric(df[phases].sum(axis=1)),
        time=lambda df: df.prod_day.dt.to_period("D"),
        time_on=lambda df: pd.to_numeric(df.time_on),
    )
    log.info(f"Read {len(df)} rows from '{table}'")

    # Print data quality report
    log.info("\nPDM data quality report (raw data, detailed)")
    log.info(report_data_quality(df, per_well=True).to_markdown(index=False))
    log.info("\nPDM data quality report (raw data, aggregated)")
    log.info(report_data_quality(df, per_well=False).to_markdown(index=False))

    # Clip values
    df = df.assign(
        production=lambda df: pd.to_numeric(df[phases].clip(lower=0.0).sum(axis=1)),
        time_on=lambda df: df["time_on"].clip(lower=0.0, upper=24.0) / 24.0,
    )

    # Deal with data quality issues. See docstring above for reasoning.
    # When time_on > 0 and production ~= 0, set time_on to 0.
    # When time_on is NaN and production ~= 0, set time_on to 0.

    df = (
        df
        # When production ~= 0, set time_on to 0.
        .assign(
            time_on=lambda df: df["time_on"].where(
                ~np.isclose(df["production"].to_numpy(), 0.0),
                0,
            )
        )
    )

    assert not df["production"].isna().any()

    # Impute time_on on every well
    dfs = []
    for _well_id, df_i in df.groupby("well_id"):
        df_i = df_i.sort_values("time")  # Interpolation needs ordering to work
        tsi = TimeSeriesInterpolator(
            time_on=df_i["time_on"].to_numpy(), production=df_i["production"].to_numpy()
        )

        time_on, _production = tsi.interpolate(maxiter=99)
        dfs.append(df_i.assign(time_on=time_on))
    df = pd.concat(dfs, ignore_index=True)

    # Sum to monthly
    df = df[["well_id", "time", "production", "time_on"]]
    if frequency == "monthly":
        log.info("Aggregating to monthly.")
        df = aggregate_production_df(df, freq="M")

    # Only keep data from before this current month/day
    freq = {"monthly": "M", "daily": "D"}[frequency]
    df = df.loc[lambda df: df.time < pd.Period.now(freq), :]

    df = filter_periods(
        df[["well_id", "time", "production", "time_on"]], period_range=period_range
    )

    log.info("\nPDM specific processing rules (in order):")
    log.info(" - on_stream_hrs is clipped to the range [0, 24]")
    log.info(f" - phases {phases!r} are clipped to the range [0, inf)")
    log.info(f" - phases {phases!r} are summed to form 'production'")
    log.info(" - (production ~= 0)                                  =>  time_on := 0")
    log.info(
        " - (on_stream_hrs IN (0, NULL)) AND (production > 0)  =>  time_on is interpolated"
    )
    log.info(" - data is summed to monthly resolution if the user asked for it\n")

    log.info("PDM data quality report (processed data, aggregated)")
    log.info(report_data_quality(df, per_well=False).to_markdown(index=False))
    return df


@validate
def load_sodir_data(
    *,
    table: str,
    phases: list[str],
    frequency: str,
    well_ids: list[str],
    period_range: list[pd.Period],
    format: str,
) -> pd.DataFrame:
    """Read data from SODIR (sokkeldirektoratet) and return DataFrame."""
    assert phases == ["oil"]
    assert frequency == "monthly"

    df = (
        load_monthly_sodir_production()
        # Select columns
        .loc[:, ["prfInformationCarrier", "prfYear", "prfMonth", "prfPrdOilNetMillSm3"]]
        # Create 'time' from year and month
        .assign(
            time=lambda df: pd.PeriodIndex.from_fields(
                year=df.prfYear, month=df.prfMonth, freq="M"
            )
        )
        # Rename well_id (which is really field id in this dataset) and production
        .rename(
            columns={
                "prfInformationCarrier": "well_id",
                "prfPrdOilNetMillSm3": "production",
            }
        )
        # Assume the well is on when production > 0
        .assign(time_on=lambda df: 1 - 1 * np.isclose(df.production.to_numpy(), 0))
        # Clip production
        .assign(production=lambda df: df.production.clip(lower=0))
        # Keep the ones we wanted
        .loc[lambda df: df.well_id.isin(well_ids), :]
    )

    return filter_periods(
        df[["well_id", "time", "production", "time_on"]], period_range=period_range
    )


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
