# Description: Postprocessing functions for the ADCA module.
import logging
import pathlib

import pandas as pd

from dca.adca.well import WellGroup

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def write_pforecast_xlsx(
    *,
    df_forecast: pd.DataFrame,
    well_group: WellGroup,
    phase: str,
    output_file: pathlib.Path,
):
    """Write pforecast import file for production potentials (Excel file)."""
    with pd.ExcelWriter(output_file) as writer:
        to_pforecast_general(well_group).to_excel(
            writer, sheet_name="General", index=False
        )
        to_pforecast_monthly(df_forecast, phase).to_excel(
            writer, sheet_name="Monthly", index=False
        )


def to_pforecast_general(wells: WellGroup) -> pd.DataFrame:
    """Prepare a dataframe with info for pForecast, General sheet."""
    return (
        pd.DataFrame([well.id for well in wells], columns=["well_id"])
        .assign(
            ProfileName=lambda df: df.well_id,
            Generic=0,
            Description=lambda df: "Profile for " + df.well_id,
            VolumeBased=1,
            AveragePE=1.0,
            DistributionType="Triangular (P10, P90)",
            UseReferenceValuesAs="Expected",
        )
        .drop(columns=["well_id"])
    )


def to_pforecast_monthly(df_forecast: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Convert the forecast DataFrame to the pForecast format.
    Note that pForecast requires units to be production/day,
    even though the period is monthly production.
    """
    cols = [
        "well_id",
        "time",
        "production",
        "forecasted_production",
        "forecasted_production_P10",
        "forecasted_production_P90",
    ]
    phase = phase.capitalize()
    return (
        df_forecast.loc[:, cols]
        .assign(
            ProfileName=lambda df: df.well_id,
            ProductionYear=lambda df: df.time.dt.year,
            ProductionMonth=lambda df: df.time.dt.month,
            # Assume we have some production p per period (month). ADCA does not
            # account for the number of days per month in any way. A month is a period of time.
            # To scale this production based on the number of days in a specific month we compute:
            # [p * daysinmonth / daysinmonth_avg]
            # Then, to scale this result to use units to production per day, we compute:
            # [p * daysinmonth / daysinmonth_avg] / daysinmonth_avg
            # = p *  daysinmonth / daysinmonth_avg**2
            # = p *  daysinmonth * monthsinday_avg**2
            # NOTE: Uncertainties (P10 / P90) DO NOT scale this way in reality.
            # The sum of P90s is not the P90s of the sum. However, it's a good enough approximation is this case.
            Potential=lambda df: df.production.fillna(df.forecasted_production)
            * df.time.dt.daysinmonth
            * (12 / 365.25) ** 2,
            PotentialLow=lambda df: df.production.fillna(df.forecasted_production_P10)
            * df.time.dt.daysinmonth
            * (12 / 365.25) ** 2,
            PotentialHigh=lambda df: df.production.fillna(df.forecasted_production_P90)
            * df.time.dt.daysinmonth
            * (12 / 365.25) ** 2,
        )
        .drop(columns=cols)
        .rename(
            columns={
                "Potential": f"{phase}Potential",
                "PotentialLow": f"{phase}PotentialLow",
                "PotentialHigh": f"{phase}PotentialHigh",
            }
        )
    )
