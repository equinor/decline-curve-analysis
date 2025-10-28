ADCA FAQ
========

This page contains frequently asked questions (FAQ) for ADCA.
Many of the questions can be answered by providing short snippets of code for pre- or postprocessing of adca inputs/outputs.


**Why do the fitted curves look bad?**

The search range for the halflife hyperparameter might be set to silly values, like ``[3, 36]`` for daily data. 
The search values are number of datapoints. 
It depends on the data, but ``[90, 1000]`` would make more sense with daily data and ``[3, 36]`` can be fine for monthly data.
In general you should carefully review the config file, and potentially set hyperparameters manually instead of tuning them.

**How many data points do I need to use ADCA?**

This is a good question, but unfortunately it's hard to give a definite answer.
It's akin to asking "how many respondents do I need to include in a survey to get good results?".
More is always better, and it depends on the desired accuracy.

Our best tip is to examine the output plots and use common sense.
With little data (a few months) you cannot expect great results, but ADCA might still be worthwhile.
Once you have a year or so you can expect good results in most cases.
The well should be on a steady, smooth decline.

**I want to use monthly frequency, but my production data has units bbl/days. How can I convert it?**

This is best done before invoking ADCA.
A Python snippet like the following should do the trick:

.. code-block:: python

   import pandas as pd
   df = pd.read_csv("production_data_daily.csv")
   df = df.assign(production=lambda df:df.production * pd.to_datetime(df.time).dt.days_in_month)
   df.to_csv("production_data_monthly.csv", index=False)


**Can you add <FEATURE/CALCULATION> to the outputs?**

It never hurts to ask, and *new outputs* can sometimes be added.
Derived information, that can be computed using existing outputs, will likely not be added.
Adding derived information tends to bloat software.

For instance, a user commented "it would be nice if in the data_report file we could also have the AVG decline."
Instead of ADCA attempting to accomodate needs like this, it's better if users compute it themselves using Excel, or with a short Python snippet like:

.. code-block:: python

   import pandas as pd
   df = pd.read_csv(r"C:\Appl\...\forecast.csv")
   df_result = (df[["well_id", "forecasted_production"]]
             # Compute difference along column
             .assign(forecast_diff=lambda df:df.forecasted_production.diff())
             # For every well ID, compute mean over differences (and count)
             .groupby("well_id").forecast_diff
             .agg([pd.Series.mean, pd.Series.count]))
   print(df_result)


**How can I view forecasts month by month?**

Engineers sometimes want to see forecasted values in a more readable format.
For instance, one row per well and one column per month.
A Pyhon snippet like the following might be useful:

.. code-block:: python

    import pandas as pd
    
    PATH = "forecast.csv"
    
    # Both are inclusive
    START = "2025-09"
    END = "2025-12"
    
    df = (pd.read_csv(PATH)
          # Pick columns
          .loc[:, ["well_id", "time", "forecasted_production"]]
          # Drop missing forecasted production (history)
          .dropna(how="any")
          # Convert daily to monthly
          .assign(time = lambda df: pd.to_datetime(df["time"]).dt.to_period("M"))
          # Filter
          .loc[lambda df:(df["time"] >= START) & (df["time"] <= END)]
          # Sum up per well and per month
          .groupby(["well_id", "time"]).sum().reset_index()
          # Pivot into wide format with wells along rows and months on cols
          .pivot(index='well_id', columns='time', values='forecasted_production')
          )
    
    print(df)
    
    df.to_csv("wide_data.csv")

   
**How can I sum forecasts?**

Every well should have history that goes up to the same period.
If so, you can use something like:

.. code-block:: python

    import pandas as pd
    path = r"forecast.csv"
    
    (pd.read_csv(path)
      # For each time period, SUM all forecasted_production
     .groupby("time")
     .forecasted_production.sum()
     
     # Only keep future periods, where we have a forecast
     .loc[lambda ser:ser > 0]
     
     # Convert to dataframe, save file
     .reset_index()
     .to_csv("summed.csv", index=False))
     
If you want to sum cumulatives, you have can use something like:

.. code-block:: python

    import pandas as pd
    path = r"forecast.csv"
    
    (pd.read_csv(path)
      # For each time period, SUM all cumulative_production (historical + forecast)
     .groupby("time")
     .cumulative_production.sum()
     
     # If you want, filter out periods that are in the future 
     .loc[lambda ser:pd.to_datetime(ser.index) > "2020-01"]
     
     # Convert to dataframe, save file
     .reset_index()
     .to_csv("summed.csv", index=False)
     )
   
   
**How can I get all a list of all well IDS for a field?**

.. code-block:: python

    from pdm_datareader import query
    
    sql = """
    SELECT DISTINCT
      WB_UWBI as well_id
    FROM PDMVW.WB_PROD_DAY
      WHERE GOV_FIELD_NAME = 'XXX'
    """
    
    # Download data
    df = query(sql)
    
    # Print every well id in .yaml syntax
    for well_id in sorted(df["well_id"]):
        print(f'    "{well_id}":')
   


**How can I fit on downhole pressure?**

Here is an example:


.. code-block:: python

    from pdm_datareader import query
    
    # YOUR WELLS GO HERE
    WELL_IDS = ["NO XXX/YYY A", "NO XXX/YYY B"]
    
    # Create SQL query
    where = f"IN {tuple(WELL_IDS)}" if len(WELL_IDS) > 1 else f"= {repr(WELL_IDS[0])}"
    sql = f"""
    SELECT 
      wb_uwbi as well_id, 
      prod_day,
      dh_press_barg as production,
      on_stream_hrs
    FROM PDMVW.WB_PROD_DAY
      WHERE wb_uwbi {where}
    """
    
    # Download data
    df = query(sql)
    
    # Clean data
    # TODO: better logic
    df = (df.assign(
            time=lambda df: df["prod_day"].dt.to_period("D"),
            production=lambda df: df["production"].clip(lower=0.0),
            time_on=lambda df: df["on_stream_hrs"].clip(lower=0.0, upper=24.0) / 24.0,
        )
        .loc[:, ["well_id", "time", "production", "time_on"]]
        )
        
    
    df.to_csv("pressure_data.csv", index=False)
    print(f"Saved {len(df)} rows to file.")
    
    # Print every well id in .yaml syntax
    for well_id in sorted(WELL_IDS):
        print(f'    "{well_id}":')
