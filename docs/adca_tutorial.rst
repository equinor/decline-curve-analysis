ADCA tutorial
=============

This tutorial assumes that you have installed ADCA and that your Python virtual environment is activated.
See :doc:`/installation` for more information about how to install ADCA.
Read :doc:`/adca` before starting this turial, or alongside it.


Creating a demo dataset
-----------------------

If you want to run on a demo dataset instead of on real-world data, you can use the ``adca init`` command.
Assuming you have followed the installation instructions, open a terminal, activate the virtual environment, change directories to a suitable folder using the ``cd`` command, then run::

  adca --help
  
To list all available sub-commands.
You'll see that one of the sub-commands is ``init``, so run the following in your terminal::

  adca init
  
Two files will be created in your current working directory.
If you open a file explorer you will see:

- ``demo.csv``: demo data with columns ``well_id``, ``time``, ``production`` and ``time_on``.
- ``demo.yaml``: a file with a basic configuration setup for ADCA.

Have a look at both files if you like.
They are text files and can be opened in Notepad on Windows.
You might have to close them before running ADCA on them if you're on Windows.

.. note::
   If you want to run on real-world wells from PDM, change ``demo.yaml`` to:
   
   .. code-block:: text

      source:
        name: "PDM"
        table: "PDMVW.WB_PROD_DAY"
        phases: ["oil"]
        frequency: "monthly"
        period_range: [null, null]
          
      wells:
        "NO 16/X-X-XX":
        "NO 16/X-X-YY":
        
   Another option is to run on the public Volve data set:
   
   .. code-block:: text

      source:
        name: "volve"
        phases: ["oil"]
        frequency: "monthly"
        
      wells:
        "NO 15/9-F-11 H":
        "NO 15/9-F-12 H":
        "NO 15/9-F-14 H":
        "NO 15/9-F-15 D":
        
   If you run on real-world wells, your numbers might differ from mine for the rest of the tutorial.
   I recommend first running on the demo dataset before running on real-world wells.


A first run
-----------

To run ADCA on the recently created files above, execute the following command in the terminal::

  adca run demo.yaml
  
Congratulations!
You have just forecasted your first group of wells!

.. note::
   Running on the demo dataset should take approximately one minute.
   Forecasting 200 real-world wells on monthly resolution will take around 12 minutes.

The result of the run are stored in the folder ``./output/YYYY-MM-DD-HH-MM-demo-demo_wells/``.
The folder is timestamped with the current time.

In the output folder you will find many files and figures:

- ``forecast.csv``: the actual forecasted values for every well - daily/monthly rates and cumulatives.
- ``curve_parameters.csv``: the curve parameters that ADCA found.
- ``log.log``: the terminal output, stored as a log file.
- Many ``.png`` figures to help you assess the fit and forecasts.

Have a look at the outputs and familiarize yourself with them.
Especially the ``.png`` figures, since they are the most important part of assessing the models.


Customizing the configuration file
----------------------------------

Because ADCA fits decline curves automatically, it removes a lot of subjective assessments.
Once you have a good setup you can re-run without much effort in the future.
Simply execute ``adca run demo.yaml`` a few months or years later.
That said, you will have to spend some time on the initial configuration.
There is no configuration that is universally the best for all forecasting scenarios, so the initial setup requires some work.

.. note::
   Need help setting up a good configuration?
   If you still have questions after reading the documentation, do not hestitate to contact us!


- The first step to building a good configuration is to decide on a few things in the configuration file ``demo.yaml``.
- The second step is to do a few trial runs and potentially change some hyper parameters.

Let us first look at the basic configuration that you should set up:

- Pick a good descriptive name. The name in the demo file is ``demo_wells``.
- Choose the correct phase, e.g. ``gas`` or ``oil``. When reading local CSV files like ``demo.csv``, this choice does not matter because the column ``production`` is read regardless of the phase specified in the config file.
- Choose the desired frequency. In general we recommend ``monthly``, but ``daily`` is possible too.
- Choose the appropriate preprocessing option, either ``producing_time`` or ``calendar_time``.
- Set ``forecast_periods`` to an appropriate number, e.g. ``360`` for a 30-year forecast on monthly resolution.

For more information about the options available in the config file, see :doc:`/adca`.

.. note::
   You can reduce the number of output plots to speed up ADCA and avoid cluttering your folders.
   You can also output more plots than the default if you want.
   To control the number of plots generated, use the ``--plot-verbosity`` argument or its shorthand ``-pv``.
   The higher the number, the more plots will be produced.
   For instance, to output all possible plots, run ``adca run demo.yaml --plot-verbosity 9``.
   To output no plots at all, run ``adca run demo.yaml --plot-verbosity 0``.
   For more information about optional arguments, run ``adca run --help``.
   
Segmenting a well
-----------------

Let us start by looking at the terminal outputs.
You can skip most of the data processing output in the beginning.
Go down to the section

.. code-block:: text

   {
     "half_life": 6.261375778652348,
     "prior_strength": 0.0011364636663857253
   }
   -------------------EVALUATE ON TEST SET WITH BEST HYPERPARAMETERS-------------------
    Negative log-likelihood: -0.4391
    RMSE in logspace: 0.2509
    Relative error (expected): 5.63%
    Relative error (P50): -15.90%
    
Here we observe that the expected case forecast, which corresponds to the column ``cumulative_production`` in ``forecast.csv``, leads to a total error of 5.63 %.
In other words, ADCA over-forecasted slightly compared to what actually happened on the test set.

Have a look at the plot ``forecast_ramp_up_well_1.png`` in the output folder.
You'll see that the decline curve does not achieve a great fit because the ramp-up period confuses the model.

You can also see this if you study table with individual errors in the ADCA terminal output.
The well ``ramp_up_well`` has a high RMSE, high relative error, etc:

.. code-block:: text

    | well_id             |   RMSE in logspace | Rel. error (expected)   | Rel. error (P50)   |
    |:--------------------|-------------------:|:------------------------|:-------------------|
    | constant_well       |             0.2191 | -1.53%                  | -13.98%            |
    | little_data_well    |             0.1787 | 9.80%                   | -14.62%            |
    | plateu_well         |             0.2606 | 9.69%                   | -2.80%             |
    | long_declining_well |             0.1079 | -7.68%                  | -18.26%            |
    | noisy_well          |             0.264  | 18.93%                  | -17.04%            |
    | ramp_up_well        |             0.463  | 36.91%                  | -37.35%            |
    | arps_well           |             0.1607 | -8.89%                  | -18.68%            |

Let us segment the problematic well by changing the ``demo.yaml`` file to::

  "ramp_up_well":
    - ["2010-09", null]
    
After making this change, you can re-run ``adca run demo.yaml``.
If you re-run without any other changes, you'll see that the output changes to:

.. code-block:: text

  Best hyperparams:
  {
    "half_life": 6.261375778652348,
    "prior_strength": 0.0011364636663857253
  }
  -------------------EVALUATE ON TEST SET WITH BEST HYPERPARAMETERS-------------------
   Negative log-likelihood: -0.4576
   RMSE in logspace: 0.1888
   Relative error (expected): 2.88%
   Relative error (P50): -14.15%


The overall test-set error went down from 5.63% to 2.88% when we segmented the well.
If you examine the plot ``forecast_ramp_up_well_1.png`` in the output folder for this new run, you'll see that the problematic ramp-up phase has been filtered out.

Hyperparameters
---------------

ADCA has a few hyperparamters.
These can either be set as fixed values, or as brackets, in which case ADCA will search for the optimal parameters within the range.
The optimal hyperparameters are those that lead to the best out-of-sample predictions on the test set.

In our runs so far, the hyperparameter ranges in ``demo.yaml`` were given as::

    # Hyperparameters given as numbers are fixed, and those given as
    # a range [low, high] are tuned out-of-sample on test data using the split
    hyperparameters:
      half_life : [6, 60]
      prior_strength : [0.001, 1]

The chosen hyperparameters were::

    Best hyperparams:
    {
      "half_life": 6.261375778652348,
      "prior_strength": 0.0011364636663857253
    }

Both parameters are in the lower end of their ranges: ``6.2613`` is near the bottom of ``[6, 60]`` and ``0.0011`` is near the bottom of ``[0.001, 1]``.
This suggests that we should broaden the search range in the lower end.
Let us try with::

    hyperparameters:
      half_life : [1, 60]
      prior_strength : [0.0000001, 0.1]
      
Making this change, saving ``demo.yaml`` and re-running adca produces:

.. code-block:: text

  Best hyperparams:
  {
    "half_life": 4.223240466094437,
    "prior_strength": 1.2915496650148853e-07
  }
  -------------------EVALUATE ON TEST SET WITH BEST HYPERPARAMETERS-------------------
   Negative log-likelihood: -0.4874
   RMSE in logspace: 0.1756
   Relative error (expected): 1.79%
   Relative error (P50): -14.30%
   
We got the error down to 1.79%.
The optimal half-life was found to be ``half_life=4.22324...``, which means that the most recent month of production data has a relative weighting of ``1``, the month before it ``2^(-1 / 4.22) = 0.84``, the month before that ``2^(-2 / 4.22) = 0.72``, and so on.

Some subjective assessment of hyperparameters can be appropriate, especially when there are few wells in the well group.
With little data, like in this demo example, there is good reason to use some engineering judgement and override the hyperparameters.
A half life of 4 months is quite short after all.

Manually setting hyperparameters to::

  hyperparameters:
    half_life : 8
    prior_strength : 0.000001 # 1e-06
  
Will make ADCA run much faster since it does not have to search for hyperparameters.
The test-set error does go up a little bit, but a half-life of 8 instead of 4 might be more appropriate.

.. note::
   You can increase or decrease the number of hyperparameter combinations used in search.
   Use the ``--hyperparam-maxfun`` argument or its shorthand ``-hm``.
   For instance, ``adca run demo.yaml --hyperparam-maxfun 10``.
   
   
Exercises
---------

Here are some exercises that are highly recommended to get some hands-on experience.

1. **Train/test split.** Set ``split`` to ``0.5`` instead of ``0.8`` in ``demo.yaml`` and examine the output figures. You should see that the split is made at 50%, so the first half of the data is in the training data set and the remaining half is in the test data set. Then try setting ``split`` to ``-1``. You should see that only the last data point is in the test data set. The train/test split should mimic your forecasting scenario - if your goal is to forecast short term, then the test set should be relatively small, and vice versa.
2. **Hyperparameter experimentation.** Manually set ``half_life=1`` and run ADCA. Look at the output figures. You should see that the curve model becomes very dependent on the last few data points. You can also set ``prior_strength=1000`` and see that every well has the same prediction, which is close to the grand mean of every well.
3. **Preprocessing options.** Read the description of ``preprocessing`` in :doc:`/adca`. Then update ``demo.csv`` with a dataset that has ``time_on=0.5``:

  .. code-block:: text
  
     well_id,time,production,time_on
     arps_well,2010-01,100,0.5
     arps_well,2010-02,100,0.5
     arps_well,2010-03,100,0.5
     arps_well,2010-04,100,0.5
     arps_well,2010-05,100,0.5
     arps_well,2010-06,100,0.5
     arps_well,2010-07,100,0.5
     arps_well,2010-08,100,0.5
   
  and run it with the preprocessing option ``producing_time``.
  Remember to remove all wells except ``arps_well`` from ``demo.yaml``.
  You should see that predictions in ``forecast.csv`` are close to ``200`` for every future period.
  Then run it with preprocessing option ``calendar_time``.
  Now you should see that predictions are close to ``100`` for every future period.