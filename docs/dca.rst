.. currentmodule:: dca.decline_curve_analysis

The DCA package
===============

The ``dca`` package contains core mathematical routines for decline curve analysis.
This part of the code is meant to be used by developers and tech-savy engineers who want to build their own DCA on top of our building blocks.
The best way to get started might be to look at the example gallery.

DCA and the improved loss function
----------------------------------

Decline Curve Analysis (DCA) is the process of fitting curves to well production time series data.
Two commonly used curves are :py:class:`Arps` and :py:class:`Exponential`.
The purpose of DCA can be to analyze past well behavior, or to predict future well behavior.

This Python package implements fast and robust DCA.
Instead of fitting a curve :math:`f(t; \boldsymbol{\theta})` to the standard least-squares loss function

.. math::
      
   \mathcal{L}(\boldsymbol{\theta}) = \sum_i \left( y_i - f(t_i; \boldsymbol{\theta}) \right)^2

we use a loss function implemented as :py:class:`CurveLoss`, which generalizes least-squares and has better out-of-sample predictions.

.. math::

   \mathcal{L}(\boldsymbol{\theta}) = 
   \sum_i w_i (\gamma) \lvert \ln y_i - \ln f(t_i; \boldsymbol{\theta}) \rvert ^p 
      + \alpha \left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right)^T 
	\boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1}  
	\left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right)
	
.. currentmodule:: dca.models
	
Taking this idea further, to a probabilistic setting where we account for autocorrelation, we implement :py:class:`AR1Model`.
This model joins an AR(1) process with a non-linear regression.
The forward simulation equations are:

.. math::

    \log(y_i) &= \log(f(t_i)) + \eta_i \\
    \eta_i &= \phi \cdot \eta_{i-1} + \sqrt{\tau_i} \cdot \epsilon_i \\
    \epsilon_i &\sim \text{GN}(\mu=0, \alpha=\sigma, \beta)


Main features of the DCA package
--------------------------------

- **Lightning fast.** Fits around 1000 wells per minute.
- **Beats least-squares.** Typically 25% reduction in forecasting RMSE compared to least squares. Measured as out-of-sample error on real well data.
- **Robustness.** Robust to outliers since we allow :math:`1 \leq p \leq 2` in the loss, as well as fitting on the log scale.
- **Proven performance.** Tested against engineering estimates on several fields, typically outperforms engineering forecasts by 20% or so.

The user must decide on the following to get the most out of the method:

- **What is the forecasting horizon?** A year? Ten years? etc.
- **What should the hyper-parameters be set to?** Decided by either manually fixing parameters, or by running cross validation on a time-based train/test split.

.. note::
   The ``dca`` package implements low-level building blocks.
   Users will have to pre-process data correctly, connect to a numerical optimizer, implement hyperparameter search if relevant, etc.
   A user can avoid all of this by using ``adca`` instead.
