"""
Main runner.
                        +----+
 config_file.yaml ----> |ADCA| ---> output files
                        +----+

             ADCA uses these objects:
            +-------------------------+
            |data_loader    WellGroup |
            |                         |
            |               Well      |
            |                         |
            |               CurveLoss |
            |                         |
            |               Arps      |
            +-------------------------+

"""

import inspect
import json
import logging
import pathlib
import pprint
import shutil
import sys
from collections.abc import Generator
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dca.adca.dataloaders import (
    load_file,
    load_PDM_data,
    load_sodir_data,
)
from dca.adca.load_yaml import yaml_safe_load
from dca.adca.postprocess import write_pforecast_xlsx
from dca.adca.utils import Bunch, clean_well_data, to_filename, to_period
from dca.adca.well import WELLGROUP_SEP, Well, WellGroup, fnum
from dca.models import Prior

# Create handler
stdout_handler = logging.StreamHandler(stream=sys.stdout)

# Create and configure a parent logger, also used in this script
log = logging.getLogger("dca")  # Children have 'dca.<__name__>' syntax
log.setLevel(logging.INFO)  # INFO or DEBUG
log.addHandler(stdout_handler)

# The warnings module redirects to the logging system
logging.captureWarnings(True)


def test_set_metrics(wellgroup, split: float):
    """Yield tuples (well:Well, metrics:dict)."""
    for well in wellgroup:
        # Weights and metrics evaluated elementwise (on each data point)
        (_, (_, _, w_test)) = well.get_train_test(split)

        # No test set data for this well
        if np.isclose(np.sum(w_test), 0):
            yield (
                well,
                {
                    "Test periods": 0,
                    "Negative log-likelihood": np.nan,
                    "RMSE in logspace": np.nan,
                    "Rel. error (expected)": np.nan,
                    "Rel. error (P50)": np.nan,
                },
            )
            continue

        neg_ll = well.negative_ll_test(split=split, p=None, sigma=None, phi=None)
        sq_errs = well.squared_log_error_test(split=split)

        # Take weighted sum of elementwise metrics
        neg_ll = np.average(neg_ll, weights=w_test)
        sq_errs = np.sqrt(np.average(sq_errs, weights=w_test))

        # Forecasting errors
        ((forecast_exp, actual_exp), (forecast_P50, actual_P50)) = list(
            well.relative_cumulative_error(split=split, q=[None, 0.5])
        )

        relative_error_expected = (forecast_exp - actual_exp) / actual_exp
        relative_error_P50 = (forecast_P50 - actual_P50) / actual_P50

        yield (
            well,
            {
                "Test periods": fnum(w_test.sum(), 2),
                "Negative log-likelihood": neg_ll,
                "RMSE in logspace": fnum(sq_errs),
                "Rel. error (expected)": format(relative_error_expected, ".2%"),
                "Rel. error (P50)": format(relative_error_P50, ".2%"),
            },
        )


class GroupProcessor:
    # TODO: Is this object even needed?

    loader_functions = {
        "PDM": load_PDM_data,
        "file": load_file,
        "sodir": load_sodir_data,
    }

    def __init__(self, *, group, output_dir):
        self.group = group
        self.output_dir = pathlib.Path(output_dir)
        assert self.output_dir.exists()
        self.errors_ = []  # Log data quality issues here

    def load_data(self) -> pd.DataFrame:
        """Load data and return DataFrame."""
        log.info("-" * 32 + "LOAD AND PROCESS RAW DATA" + "-" * 32)

        group = self.group
        loader_func = self.loader_functions[group.source.name]

        table = None if group.source.name == "sodir" else group.source.table
        format = group.source.format if hasattr(group.source, "format") else None
        period_range = (
            [None, None]
            if not hasattr(group.source, "period_range")
            else group.source.period_range
        )

        # Unpack wells or group of wells, e.g. ["A", "B,C"] => ["A", "B", "C"]
        well_ids = [
            well
            for well_or_group in group.wells
            for well in well_or_group.split(WELLGROUP_SEP)
        ]

        df = loader_func(
            table=table,
            phases=group.source.phases,
            frequency=group.source.frequency,
            well_ids=well_ids,
            period_range=period_range,
            format=format,
        )

        log.info(f"Loaded {df.well_id.nunique()} wells and {len(df)} rows.")

        return df

    def report(self, df):
        """Return a summary dataframe."""

        agg = {
            "well_id": pd.Series.count,
            "time_on": pd.Series.mean,
            "production": pd.Series.mean,
        }

        df_summary_all = (
            df.groupby("well_id")
            .agg(agg)
            .rename(
                columns={
                    "well_id": "count",
                    "time_on": "AVG(time_on)",
                    "production": "AVG(prod)",
                }
            )
        )
        df_summary_pp = (
            df[(df["production"] > 0) & (~np.isclose(df["production"].to_numpy(), 0.0))]
            .groupby("well_id")
            .agg(agg)
            .rename(
                columns={
                    "well_id": "count | (prod > 0)",
                    "time_on": "AVG(time_on) | (prod > 0)",
                    "production": "AVG(prod) | (prod > 0)",
                }
            )
        )

        return (
            df_summary_all.merge(
                df_summary_pp, left_index=True, right_index=True, how="outer"
            )
            .sort_values("AVG(time_on) | (prod > 0)")
            .round(2)
            .reset_index()
        )

    def plot_raw_df(self, raw_df: pd.DataFrame) -> None:
        """Loop over each well_id and plots raw data."""

        for well_id, df_well in raw_df.groupby("well_id"):
            df_well = df_well.sort_values("time")
            time = pd.to_datetime(df_well.time.dt.to_timestamp())

            # Plot using datetimes (before any cleaning and splitting)
            fig, ax = plt.subplots()
            ax.set_title(f"Well '{well_id}' (raw data)")
            ax.set_ylabel("Production")
            ax.plot(time, df_well.production)
            ax.grid(True, ls="--", zorder=0, alpha=0.33)
            fig.tight_layout()
            fig.savefig(
                self.output_dir / f"raw_{well_id.replace('/', '')}.png", dpi=200
            )
            plt.close()

    def plot_wellgroup(self, wellgroup: WellGroup) -> None:
        """Plot processed data for each well in the wellgroup."""
        for well in wellgroup:
            fig, _ax = well.plot(split=1.0, logscale=True)
            fig.savefig(
                self.output_dir
                / f"cleaned_{well.id.replace('/', '')}_{well.segment}.png",
                dpi=200,
            )
            plt.close()

    def df_to_wellgroup(self, df: pd.DataFrame) -> WellGroup:
        """Read groups from DataFrame, return a WellGroup object."""
        log.info("-" * 32 + "GENERAL DATA CLEANING AND SEGMENT SPLITTING" + "-" * 32)

        group = self.group

        # Convert each well time series to a Well object
        wells = WellGroup()

        for well_id, df_well in df.groupby("well_id"):
            log.info(f"Parsing data for well '{well_id}'")
            df_well = df_well.sort_values("time")

            # Clean the data and report any data quality issues
            cleaned, errors = clean_well_data(
                production=df_well.production.to_numpy(),
                time_on=df_well.time_on.to_numpy(),
                time=df_well.time,
                return_log=True,
            )
            self.errors_.extend(errors)

            # If there are data quality issues, report them here
            df_qa = pd.DataFrame(errors, columns=["condition", "action", "count"])
            if set(df_qa.action) == {"unchanged"}:
                log.info(f"No data quality issues with '{well_id}'.")
            else:
                log.warning(f"Data quality report for '{well_id}':")
                log.warning(df_qa.to_markdown(index=False))
            log.info("")

            # Create a well with clean data
            well = Well(
                id=well_id,
                time=cleaned.time,
                production=cleaned.production,
                time_on=cleaned.time_on,
                # Common parameters
                preprocessing=group.preprocessing,
                curve_model=group.curve_fitting.curve_model,
            )
            wells.append(well)

        return wells

    def aggregate_wellgroup(self, wellgroup):
        """Aggregate a WellGroup object using information from the group."""

        # Aggregate the wells (sum production, average time_on)
        for well_id_or_group in self.group.wells:
            if WELLGROUP_SEP in well_id_or_group:  # Only sum groups
                log.info(f" Summing wells: {well_id_or_group}")
                wellgroup = wellgroup.aggregate(
                    group_ids=well_id_or_group.split(WELLGROUP_SEP)
                )

        return wellgroup

    def segment_wellgroup(self, wellgroup):
        """Segment a WellGroup object using information from the group."""

        # Segment the wells
        segmented_wells = WellGroup()
        for well in wellgroup:
            segments = self.group.wells[well.id]

            # log.info(f" Well '{well_id}' has segments: {segments}")
            for well_segment in well.split(segments):
                seg, seglength = well_segment.segment, len(well_segment)

                t_min, t_max = well_segment.time.min(), well_segment.time.max()
                log.info(
                    f" Created segment {seg} of length {seglength}: [{t_min}, {t_max}]"
                )

                # Clean and add
                if len(well_segment) == 0:
                    msg = f"  Zero length segment. Removing {well_segment}"
                    log.info(msg)
                    continue

                segmented_wells.append(well_segment)
            log.info("")

        return segmented_wells


def read_config(file_path: str) -> Generator[dict, None, None]:
    """Read a YAML config file, yielding a Bunch (dict) for each group."""
    file_path = str(file_path)
    assert file_path.endswith(".yaml")

    with open(file_path, encoding="utf-8") as file_handle:
        # Process each group in turn
        for group in yaml_safe_load(file_handle):
            group = Bunch(**group)  # A dictionary with attribute-lookups
            freq = {"monthly": "M", "daily": "D"}[group.source.frequency]

            # Convert dates to Periods in the 'period_range'
            if hasattr(group.source, "period_range"):
                group.source["period_range"] = [
                    to_period(period, freq=freq)
                    for period in group.source["period_range"]
                ]

            # Convert dates to Periods in segments for each well
            for well_id, segments in group.wells.items():
                if not segments:
                    continue

                group.wells[well_id] = [
                    (
                        to_period(start, freq=freq),
                        to_period(end, freq=freq),
                    )
                    for (start, end) in segments
                ]

            # If the split is a period, e.g. "2020-01", then convert it
            # Integers and floats are converted automatically by pyyaml
            if isinstance(group.curve_fitting.split, str):
                group.curve_fitting["split"] = to_period(
                    group.curve_fitting.split, freq=freq
                )

            # If the 'forecast_periods' is a period, e.g. "2020-01", then convert it
            # Integers are converted automatically by pyyaml
            if isinstance(group.curve_fitting.forecast_periods, str):
                group.curve_fitting["forecast_periods"] = to_period(
                    group.curve_fitting.forecast_periods, freq=freq
                )

            yield group


def save_well_plots(
    *,
    output_prefix: str,
    output_dir: pathlib.Path,
    wells: WellGroup,
    split: float = 1.0,
    logscale: bool = False,
    plot_type: str = "production",
) -> None:
    """For every well in the wellgroup, plot and save them."""
    log.info(f"Writing plots with prefix: {output_prefix!r}")
    assert plot_type in ("production", "cumulative", "simulations")
    for well in wells:
        q = (0.1, 0.5, 0.9)
        if plot_type == "cumulative":
            fig, _ = well.plot_cumulative(split=split, logscale=logscale, q=q)
        elif plot_type == "production":
            fig, _ = well.plot(split=split, logscale=logscale, q=q)
        elif plot_type == "simulations":
            fig, _ = well.plot_simulations(
                split=split,
                logscale=logscale,
                simulations=5,
                q=q,
            )
        else:
            raise ValueError(f"Wrong plot type: {plot_type}")

        filename = to_filename(well.id.split(",")).replace("/", "")

        fig.savefig(
            output_dir / f"{output_prefix}_{filename}_{well.segment}.png",
            dpi=200,
        )
        plt.close()


def median_params(wells):
    """Compute the weighted median of all parameters optimized over."""
    param_names = list(wells[0].parameters_.keys())
    weights = [w.get_params()["total_time_on"] for w in wells]
    params = {
        n: np.quantile(
            [w.parameters_[n] for w in wells],
            weights=weights,
            q=0.5,
            axis=0,
            method="inverted_cdf",
        )
        for n in param_names
    }
    return {
        k: float(v) if v.ndim == 0 else tuple(float(v_j) for v_j in v)
        for (k, v) in params.items()
    }


def process_file(
    *,
    config_path,
    current_directory,
    current_time=None,
    hyperparam_maxfun=25,
    plot_verbosity=1,
):
    """Process a single file. Read the file, run DCA, save figures/results.

    Parameters
    ----------
    config_path : pathlib.Path
        A Path object pointing to a .yaml file.
    current_directory : pathlib.Path
        The directory where the 'output' folder will be created.
    current_time : str, optional
        String representing current time, e.g. '2024-07-09-07-03'.
    hyperparam_maxfun : int, optional
        Number of hyperparameter iterations. The default is 25.
    """
    assert isinstance(config_path, pathlib.Path)
    assert isinstance(current_directory, pathlib.Path)
    assert config_path.suffix == ".yaml"
    assert isinstance(hyperparam_maxfun, int)
    assert hyperparam_maxfun > 0

    if current_time is None:
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M")

    # Setting this will cause all imported modules (DCA code) to raise a
    # FloatingPointError if something fails. This is very conservative,
    # but we prefer the program to terminate if there are overflows.
    # In practice this typically happens in hyperparameter search if values
    # are too large/small, and does not affect the final hyperparameters.
    np_errstate = np.geterr()
    np.seterr(all="raise", under="warn")  # "warn" or "raise"

    # Read the config
    groups = read_config(config_path)

    for group in groups:
        # Remove all existing handlers from the log
        while log.hasHandlers():
            try:
                log.removeHandler(log.handlers[0])
            except IndexError:
                break

        # Create output directory for files/logs/figures
        output_dir = pathlib.Path(
            current_directory
            / "output"
            / f"{current_time}-{config_path.stem}-{group.name}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Created output directory: {output_dir}")
        shutil.copy(config_path, output_dir)  # Copy config file

        # Set up handler in the output directory
        file_handler = logging.FileHandler(
            filename=output_dir / "log.log", mode="w", encoding="utf-8"
        )
        log.addHandler(file_handler)
        log.addHandler(stdout_handler)
        log.setLevel(logging.INFO)

        log.info(f"Running ADCA (version: {version('dca')})")
        log.info(
            f"Read file: '{config_path}' \nGroup:\n{pprint.pformat(group, indent=2)}"
        )
        group_processor = GroupProcessor(group=group, output_dir=output_dir)

        # Load DataFrame from the data source
        # ---------------------------------------------------------------------
        df = group_processor.load_data()

        # Report statistics for each well
        # ---------------------------------------------------------------------
        log.info("-" * 32 + "DATA SUMMARY PER WELL" + "-" * 32)

        df_summary = group_processor.report(df)
        log.info(df_summary.to_markdown(index=False))
        df_summary.to_csv(output_dir / "data_report.csv", index=False)
        log.info(f"Wrote summary to: {output_dir / 'data_report.csv'}")

        # Split segments into Well objects, clean them and return as WellGroup
        # ---------------------------------------------------------------------

        # Plot raw data before converting DF to WellGroup
        if plot_verbosity >= 1:
            group_processor.plot_raw_df(raw_df=df)

        wells = group_processor.df_to_wellgroup(df=df)
        wells = group_processor.aggregate_wellgroup(wells)
        wells = group_processor.segment_wellgroup(wells)

        n_thetas = len(inspect.Signature.from_callable(wells[0].curve_model).parameters)

        # Plot processed data after converting DF to WellGroup
        if plot_verbosity >= 2:
            group_processor.plot_wellgroup(wells)

        # Summarize data quality issues
        # ---------------------------------------------------------------------
        log.info("-" * 32 + "SUMMARY OF DATA QUALITY" + "-" * 32)
        df_errors = pd.DataFrame(
            group_processor.errors_, columns=["condition", "action", "count"]
        )
        df_errors = (
            df_errors.groupby(["condition", "action"])
            .sum()
            .reset_index()
            .assign(perc=lambda df: (df["count"] / df["count"].sum() * 100).round(1))
        )
        log.info(df_errors.to_markdown())

        # Fit the prior using the pilot estimate strategy
        # ---------------------------------------------------------------------
        neg_logpdf_pilot = Prior(theta_mean=np.zeros(n_thetas))

        log.info("-" * 32 + "PILOT ESTIMATE" + "-" * 32)
        half_life = max(len(well) for well in wells)
        prior_strength = 0.01
        msg = f"Pilot estimate. {prior_strength=} {half_life=}"
        log.info(msg)
        wells_prior = wells.fit(
            half_life=half_life,
            prior_strength=prior_strength,
            split=group.curve_fitting.split,
            neg_logpdf_prior=neg_logpdf_pilot,
        )

        prior_params = median_params(wells_prior)
        PRIOR_THETA = np.array(prior_params["theta"])
        log.info(f"Posterior theta (after pilot estimate): {PRIOR_THETA}")
        log.info(
            f"Posterior params (after pilot estimate):\n{json.dumps(prior_params, indent=2)}"
        )

        # Tune hyperparameters
        # ---------------------------------------------------------------------
        log.info("-" * 32 + "TUNE HYPERPARAMETERS" + "-" * 32)
        log.info(f"Train/test split: {group.curve_fitting.split}")
        # Param choices were found by some rough tuning on data
        phi_v_b = 13.48 if group.source.frequency == "monthly" else 282.89
        neg_logpdf_prior = Prior(theta_mean=np.array(PRIOR_THETA), phi_v_b=phi_v_b)

        best_hyperparameters = wells.tune_hyperparameters(
            half_life=group.hyperparameters.half_life,
            prior_strength=group.hyperparameters.prior_strength,
            # Default is to fit on all data
            split=group.curve_fitting.split,
            neg_logpdf_prior=neg_logpdf_prior,
            # Number of DIRECT calls
            hyperparam_maxfun=hyperparam_maxfun,
        )
        log.info(f"Best hyperparams:\n{json.dumps(best_hyperparameters, indent=2)}")

        # Fit on the test/train split, plot and evaluate test set performance
        # ---------------------------------------------------------------------
        log.info("-" * 32 + "EVALUATE ON TEST SET WITH BEST HYPERPARAMETERS" + "-" * 32)

        # Fit using the best found hyperparameters
        wells = wells.fit(
            **best_hyperparameters,
            split=group.curve_fitting.split,
            neg_logpdf_prior=neg_logpdf_prior,
        )

        # Evaluate log-loss, RMSE in logspace, and relative errors in forecasts
        log_loss = wells.score(
            split=group.curve_fitting.split,
            p=None,
            sigma=None,
            phi=None,
        )
        rmse = wells.rmse_log(split=group.curve_fitting.split)
        relative_error_expected, relative_error_P50 = list(
            wells.relative_cumulative_error(
                split=group.curve_fitting.split, q=[None, 0.5]
            )
        )

        log.info(f" Negative log-likelihood: {fnum(log_loss)}")
        log.info(f" RMSE in logspace: {fnum(rmse)}")
        log.info(f" Relative error (expected): {relative_error_expected:.2%}")
        log.info(f" Relative error (P50): {relative_error_P50:.2%}")

        # Output various train/test splitted figures
        if plot_verbosity >= 1:
            log.info("Writing debugging plots (.png files) on test set.")
            save_well_plots(
                output_prefix="split_test",
                output_dir=output_dir,
                wells=wells,
                split=group.curve_fitting.split,
                logscale=True,
            )

            save_well_plots(
                output_prefix="split_test_cumulative_nolog",
                output_dir=output_dir,
                wells=wells,
                split=group.curve_fitting.split,
                logscale=False,
                plot_type="cumulative",
            )

        if plot_verbosity >= 2:
            save_well_plots(
                output_prefix="split_test_nolog",
                output_dir=output_dir,
                wells=wells,
                split=group.curve_fitting.split,
                logscale=False,
            )

            save_well_plots(
                output_prefix="split_test_cumulative",
                output_dir=output_dir,
                wells=wells,
                split=group.curve_fitting.split,
                logscale=True,
                plot_type="cumulative",
            )

        if plot_verbosity >= 3:
            save_well_plots(
                output_prefix="split_test_simulations_nolog",
                output_dir=output_dir,
                wells=wells,
                split=group.curve_fitting.split,
                logscale=False,
                plot_type="simulations",
            )
            save_well_plots(
                output_prefix="split_test_simulations",
                output_dir=output_dir,
                wells=wells,
                split=group.curve_fitting.split,
                logscale=True,
                plot_type="simulations",
            )

        # Report test set errors on every well - sorted by error
        log.info("Printing test set errors on every well.")
        log.info(" - Test periods => number of periods (days/months) in test set")
        log.info(" - Negative log-likelihood => model fit (lower is better)")
        log.info(" - RMSE in logspace => root mean squared error (lower is better)")
        log.info(
            " - Rel. error (expected) => relative error in cumulative forecast using expected (closer to 0% is better)"
        )
        log.info(
            " - Rel. error (P50) => relative error in cumulative forecast using P50 (closer to 0% is better)"
        )

        wells_scores = [
            ({"well_id": well.id, "segment": well.segment} | metrics)
            for (well, metrics) in test_set_metrics(
                wells, split=group.curve_fitting.split
            )
        ]

        df_scores = (
            pd.DataFrame.from_records(wells_scores)
            .sort_values("Negative log-likelihood")
            .assign(
                **{
                    "Negative log-likelihood": lambda df: df[
                        "Negative log-likelihood"
                    ].map(fnum)
                }
            )
        )
        log.info(df_scores.to_markdown(index=False))

        # Fit on all data
        # ---------------------------------------------------------------------
        log.info("-" * 32 + "FIT ON ALL DATA" + "-" * 32)
        log.info(f"Fitting with prior mean and params\n{best_hyperparameters}")
        wells = wells.fit(
            **best_hyperparameters,
            split=1.0,  # Fit on all data => split=1.0
            neg_logpdf_prior=neg_logpdf_prior,
        )

        if plot_verbosity >= 1:
            log.info("Writing debugging plots (.png files).")
            save_well_plots(
                output_prefix="forecast",
                output_dir=output_dir,
                wells=wells,
                logscale=True,
            )

        if plot_verbosity >= 2:
            save_well_plots(
                output_prefix="forecast_nolog", output_dir=output_dir, wells=wells
            )
            save_well_plots(
                output_prefix="forecast_cumulative_nolog",
                output_dir=output_dir,
                wells=wells,
                plot_type="cumulative",
            )
            save_well_plots(
                output_prefix="forecast_cumulative",
                output_dir=output_dir,
                wells=wells,
                logscale=True,
                plot_type="cumulative",
            )

        if plot_verbosity >= 3:
            save_well_plots(
                output_prefix="forecast_simulations_nolog",
                output_dir=output_dir,
                wells=wells,
                logscale=False,
                plot_type="simulations",
            )
            save_well_plots(
                output_prefix="forecast_simulations",
                output_dir=output_dir,
                wells=wells,
                logscale=True,
                plot_type="simulations",
            )

        # Forecast and save DF
        # ---------------------------------------------------------------------
        log.info("-" * 32 + "WRITE FORECAST TO FILE" + "-" * 32)
        simulations = 1_000  # 1000 => ~ 70 MB of memory for 25 years @ daily
        log.info(f"Each forecast uses {simulations} simulations")
        df_forecast = wells.to_df(
            forecast_periods=group.curve_fitting.forecast_periods,
            q=[0.1, 0.5, 0.9],
            simulations=simulations,
        )
        # TODO: All segments are forecasted. Only keep the last segment?
        df_forecast.to_csv(output_dir / "forecast.csv", index=False)
        msg = f"Wrote {len(df_forecast)} rows for {len(wells)}"
        msg += f" wells to file: {output_dir / 'forecast.csv'}"
        log.info(msg)

        if "postprocessing" in group and "pforecast" in group.postprocessing:
            # Save for pForecast import
            # TODO: handle "oil, water" as liquid
            # TODO: handle daily -> monthly
            if group.source.frequency == "monthly" and len(group.source.phases) == 1:
                write_pforecast_xlsx(
                    df_forecast=df_forecast,
                    well_group=wells,
                    phase=group.source.phases[0],
                    output_file=output_dir / "pforecast.xlsx",
                )
                msg = f"Wrote {len(df_forecast)} rows for {len(wells)}"
                msg += f" wells to file: {output_dir / 'pforecast.xlsx'}"
                log.info(msg)
            else:
                msg = "Skipping pForecast Excel file since we don't have monthly data."
                log.info(msg)

        # Output curve parameters
        # ---------------------------------------------------------------------
        log.info("-" * 32 + "WRITE ARPS CURVE PARAMETERS TO FILE" + "-" * 32)

        df_params = wells.get_params(q=[0.1, 0.5, 0.9])
        df_params.to_csv(output_dir / "curve_parameters.csv", index=False)
        log.info(
            f"Wrote curve parameters to file: {output_dir / 'curve_parameters.csv'}"
        )

        # Set back error state to what it was
        np.seterr(**np_errstate)


if __name__ == "__main__":
    # The current directory and the current time
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
    NOW = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M")

    # Create an iterable over paths and groups
    file_paths = (PROJECT_ROOT / "well_group_examples").glob("*.yaml")

    # Read config files and loop over all groups
    for file_path in file_paths:
        if "config" in str(file_path):
            continue

        if "20well" not in str(file_path):
            continue

        process_file(
            config_path=file_path, current_directory=PROJECT_ROOT, current_time=NOW
        )
