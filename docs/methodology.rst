.. currentmodule:: dca.decline_curve_analysis

Methodology
===========

Theory
------

DCA is the process of fitting parametric curves to well production time series data.
The purpose is either to (1) analyze past well behavior, or to (2) predict future well behavior.
This project focuses on predicting future production.

The most commonly used curve is the three-parameter :py:class:`Arps` equation, which is an extension of the exponential function :py:class:`Exponential` that permits sub-exponential decline under certain parameter choices.
The most commonly used loss function is least squares, but adjusting this loss function leads to better results.

Results
-------

The main benefit from using our DCA method with the ADCA system is:

- **Speed.** We fit thousands of wells per minute.
- **Consistency.** No subjective assessments of curve parameters needed.
- **Accuracy.** In every test we've performed, we attain better predictions than manual DCA curve fitting.

More details on accuracy:

- **15000 on-shore gas wells**
  - Our generalization of the loss function lead to a 25% reduction in forecasting error, compared to using least-squares loss.
  - Furthermore, we achieved 20% better forecasts (lower RMSE) than DCA performed by engineers. This was measured using RMSE. On around 75% of the wells, our method outperformed engineering estimates.
- **NCS fields**
  - We have done comparisons on NSC (Norwegian Continental Shelf) fields. Here there is typically less data and access to historical forecasts are more scarce. However, we found our method to outperform engineering estimates of DCA parameters in every case. Due to the low number of wells we hesitate to claim statistical significance.

Assumptions
-----------

DCA operates under the assumption that wells exhibits a steady rate of production decline.
This is a common scenario for mature wells that have passed the peak production phase and entered a more predictable decline phase.
Deviations from this assumption may impact the accuracy of the model's predictions.
Use common sense and look at the debugging figures that the ``adca`` system outputs to assess performance.

Limitations
-----------

The limitations are the same as in any DCA method.
We require consistent and uninterrupted decline in production. 
DCA does not capture the complexities of well performance that can arise from operational interventions, reservoir heterogeneities, or other geological events.

Users should apply DCA within the appropriate context and adhere to the following guidelines:

- Ensure the production data is complete and accurate to maintain the integrity of the forecasts.
- Recognize the steady decline assumption and consider its implications for the applicability of the model to specific wells.
- Utilize the DCA tool in conjunction with engineering judgment.

Uncertainties
-------------

We do not provide any low/high case or uncertainty.

Guidelines
----------

Users should adhere to the following guidelines:

- Ensure the production data is complete and accurate to maintain the integrity of the forecasts.
- Recognize the steady decline assumption and consider its implications for the applicability of the model to specific wells.
- Utilize the DCA tool in conjunction with engineering judgment.


Links and references
--------------------

- :doc:`dca_report`	      


Relevant papers on DCA (stars denote the most relevant):

- ⭐Arps, J.J. (1945). "Analysis of Decline Curves". Transactions of the AIME, 160(01), 228–247. DOI: 10.2118/945228-G.
- ⭐Lee, Se Yoon and Mallick, Bani. (2021). "Bayesian Hierarchical Modeling: Application Towards Production Results in the Eagle Ford Shale of South Texas". Sankhya B. DOI: 10.1007/s13571-020-00245-8.
- Jochen, V. A. and Spivey, J. P. (1996). "Probabilistic Reserves Estimation Using Decline Curve Analysis with the Bootstrap Method". OnePetro. DOI: 10.2118/36633-MS.
- Tang, Huiying et al. (2021). "A novel decline curve regression procedure for analyzing shale gas production". Journal of Natural Gas Science and Engineering, 88, 103818. DOI: https://doi.org/10.1016/j.jngse.2021.103818.
- Tan, Lei et al. (2018). "Methods of Decline Curve Analysis for Shale Gas Reservoirs". Energies, 11(3), 552. DOI: 10.3390/en11030552.

Books that are somewhat relevant (loss functions, statistics, optimization):

- Bishop, Christopher M. (2011). Pattern Recognition and Machine Learning. New York: Springer. ISBN: 978-0-387-31073-2.
- McElreath, Richard. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and STAN (2nd Edition). Boca Raton: Chapman and Hall/CRC. ISBN: 978-0-367-13991-9.
- Gelman, Andrew et al. (2013). Bayesian Data Analysis (3rd Edition). Boca Raton: Chapman and Hall/CRC. ISBN: 978-1-4398-4095-5.
- Boyd, Stephen and Vandenberghe, Lieven. (2004). Convex Optimization. Cambridge University Press.

