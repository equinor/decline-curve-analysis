
Introduction
============


The `decline-curve-analysis repository <https://github.com/equinor/decline-curve-analysis>`_ contains code and theory for automated and improved curve fitting for Decline Curve Analysis (DCA).
The product provides automation tools for Reservoir Engineers, Production Engineers or others to do efficient, high quality DCA.

In addition to efficiency gains, the AutoDCA curve fitting model is seen to increase accuracy in forecasting where a decline curve approach is appropriate.
Intended use is for well or field forecasts where the wells or field is in a medium to tail production phase or the production history shows the wells to be on decline or gave sufficient data to make a trend.
The use is for long term and short term forecasting routines, i.e.,  `RNB <https://www.sodir.no/en/regulations/reporting_and_applications/revised-national-budget/>`_ and the Finance and Control annual cycle.

**Two Python packages are provided:**
   
- ``dca``: core mathematical routines (decline curve functions, loss functions, time series pre-processing, etc.)
- ``adca``: automatic DCA system (data loaders, config file readers, plotting, logging, output handling, pre-preprocessing, hyperparameter tuning, etc.)

The ``adca`` system wraps the building blocks found in ``dca`` into a feature-complete and more user friendly system that automatically performs DCA on a set of local files or a data source such as the Production Data Mart (PDM).

.. note:: 

   Do you want to use the automatic DCA system? 
   Contact Knut Utne Hollund ``kuho@equinor.com`` or Tommy Odland ``todl@equinor.com`` to get started.
