.. currentmodule:: dca.decline_curve_analysis

DCA - Technical report
======================

.. contents::
   :local:
   
.. plot:: plots/plot_dca_intro.py
   :show-source-link: True
   
.. note::
   This report is more static in nature than the rest of the documentation.
   Always check the API documentation, docstrings and code for details.

Abstract
--------

Decline Curve Analysis (DCA) is the process of fitting curves to well production time series data.
The purpose of DCA is typically to either (1) analyze past well behavior or (2) to predict future production (forecasting).
The most commonly used curve is the three-parameter :py:class:`Arps` equation. 
Arps is an extension of the :py:class:`Exponential` equation that permits sub-exponential decline under certain parameter choices.
The most commonly used loss function is least squares loss.

In this report we discuss our experiences after years of working with DCA for on-shore gas and off-shore oil and gas wells.
We state useful analytical properties of the :py:class:`Arps` equation, give practical tips for preprocessing data for DCA, and generalize the least squares loss function to :py:class:`CurveLoss`.

Introduction and previous work
------------------------------

Decline Curve Analysis (DCA) dates back to the start of the 1900s, and the history is detailed in [ARP45]_.
Jan J. Arps proposed fitting observed production to a curve of the form

.. math::
   \mu(t; q, h, D) = q (1 + h D t)^{-1/h},

which we will refer to as the Arps equation.

.. plot:: plots/plot_dca_parameters.py
   :show-source-link: True

Fitting in the 1940s was done by graphical methods on paper or by algorithms specific to the curve being fit.
These days we use general purpose algorithms for minimizing loss functions.

Many recent papers describe DCA and suggest marginal improvements, but overall the core methodology is unchanging.
For instance, [TANG21]_ propose taking the logarithm of the production rate before fitting a decline curve.
[LIU18]_ list eight different DCA models, e.g. Arps, power law, stretched exponential, and so forth.
A non-quantitative comparison is provided.
[LIANG20]_ compare 18 different DCA models.
Wells are classified as one of three different types based on first year performance, and for each type of well a different recommendation is given.
Unfortunately no quantitative error metrics are given.
Other papers are [JOCH96]_.

The papers above are published.
However, when searching the internet one finds many low-quality reports concerning DCA.
They are typically all over the place in terms of methodology, have few or no quantitative error metrics, contain pixelated or unintelligible figures, are not proofread, have not been peer-reviewed, and so forth.

In our view the exact form of the DCA model is not as important as (1) the loss function, (2) the data quality and (3) the data preprocessing.
We also believe that while examining individual wells is worthwhile, such an analysis must be accompanied by a birds-eye view of performance over many wells.
It is always possible to find individual wells where performance is particularly good or bad, and we must assess overall error in our analyses.

Decline Curves
--------------

This section contains information about decline curves.
Any curve :math:`f(t; \boldsymbol{\theta})` that is (1) positive, (2) decreasing and (3) goes to zero is a candidate curve for use in DCA.
One way to find such curves is to consider probability density functions (PDFs).
In fact, both the Exponential curve and the Arps curve are of this form: 

- the Exponential curve is the PDF of the `exponential distribution <https://en.wikipedia.org/wiki/Exponential_distribution>`_
- the Arps curve is the PDF of the `generalized Pareto distribution <https://en.wikipedia.org/wiki/Generalized_Pareto_distribution>`_

The Exponential curve
^^^^^^^^^^^^^^^^^^^^^

The simplest decline curve is the exponential curve :math:`f(t; C, k) = C \exp(-kt)` with parameters constrained to be positive.

A re-parametrization that leads to parameters that are not constrained to be positive, and are generally on the same order of magnitude, is:

.. math::
   \theta_1 &= log(C / k)  \qquad & &C= \exp(\theta_1 - \theta_2) \\
   \theta_2 &= -log(k)     \qquad & &k= \exp(-\theta_2)

This leads to the following equation:

.. math::

   \log \left( f(t; \theta_1, \theta_2) \right) = \theta_1 - \theta_2 - t \exp(-\theta_2)

Integrals, gradients and other mathematical properties are simple to derive, and will not be presented here (see the code and the class :py:class:`Exponential`).
Do notice that :math:`\int_0^{\infty} f(t; C, k) \, dt = C/k = \exp(\theta_2)`, so the logarithm of the integral is equal to :math:`\theta_1`.
This gives the parameter :math:`\theta_1` a nice interpretation.
The parametrization above also has property that :math:`\lim_{\theta_3 \to -\infty} \text{arps}(\theta_1, \theta_2, \theta_3) = \text{exponential}(\theta_1, \theta_2)`.

The Arps curve
^^^^^^^^^^^^^^

.. plot:: plots/plot_arps_and_exponential.py
   :show-source-link: True
   
   The Exponential curve and Arps curve for various values of :math:`\theta_3`.
   As :math:`\theta_3` goes to negative infinity, the Arps curve converges to the Exponential curve.

Following the notation introduced by [LEE21]_, the Arps curve is given by

.. math::
   f(t; q, h, D) = q \left(1 + hDt \right)^{-1/h}.

The parameters :math:`q` and :math:`D` are constrained to be positive.
If :math:`0 < h < 1`, then :math:`\int_0^{\infty} f(t) \, dt` is finite.
If :math:`\lim_{h \to 0}`, then :math:`\left(1 + hDt \right)^{-1/h} = \exp \left( -Dt \right)`.
This follows by from one from the definitions of :math:`\exp (x)`, namely :math:`\exp x = \lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n`.

Integrals
"""""""""

Since :math:`f(t; q, h, D)` is fit to a time series of production rates (e.g. production per day), it's useful to be able to integrate the Arps curve to obtain total production up to a certain point in time.

Some useful integral relationships are:

.. math::
   \int f(t; q, h, D) \, dt &= \frac{q \left( 1 + hDt \right)^{\frac{h-1}{h}}}{(h-1)D} + C \\
   \int_0^T f(t; q, h, D) \, dt &= \frac{q \left[  (1 + ThD)^{\frac{h-1}{h}} - 1 \right]}{(h-1) D} \\
   \int_T^\infty f(t; q, h, D) \, dt &= \frac{q}{(1-h) D (1 + T h D)^{\frac{1-h}{h}}} \\
   \int_0^\infty f(t; q, h, D) \, dt &= \frac{q}{D(1-h)}

More integrals, gradients and transformations are implemented in the code, see :py:class:`Arps`.

Reparametrization
"""""""""""""""""

As shown in [LEE21]_, the Arps curve is identical to the (scaled) `generalized Pareto distribution <https://en.wikipedia.org/wiki/Generalized_Pareto_distribution>`_ after a reparametrization:

.. math::
   \theta_1 &= \log \left(   q  / ((1 - h) D) \right) \qquad  & & h= 1 / (1 + \exp(-\theta_3)) \\
   \theta_2 &= \log \left( 1  / ((1 - h) D) \right)   \qquad & & D= 1 / ((1 - h) \exp(\theta_2) ) \\
   \theta_3 &= \log \left(  h  / (1 - h) \right)      \qquad & & q= \exp(\theta_1) (1-h ) D

When we apply the transformation, we obtain the logarithm of the Arps curve as

.. math::
   \mu(t; \boldsymbol{\theta}) =\theta_1 - \theta_2 - \left( 1 + e^{-\theta_3} \right) \log \left( 1 + t e^{\theta_3 - \theta_2} \right),

which is Equation (3.3) in [LEE21]_.
Notice that :math:`\theta_1` determines the logarithm of the integral: :math:`\exp \theta_1 = q  / ((1 - h) D) = \int_0^\infty f(t; q, h, D) \, dt`.
Also note that thinking in terms of scaled probability distributions for DCA opens up an interesting possibility: *any* sensible probability density function can be scaled and used for DCA, e.g. the Weibull distribution or the Gamma distribution.

Working in :math:`(\theta_1, \theta_2, \theta_3)`-space instead of :math:`(q, h, D)`-space has some practical advantages:

* **Parameters on the same scale.**
  In :math:`\boldsymbol{\theta}`-space all parameters are on the same order of magnitude.
  This will help an optimizer.
  On the original scale :math:`q` is typically large, while :math:`h \approx 1`.

* **Parameters are unbounded.**
  In :math:`\boldsymbol{\theta}`-space all parameters are unbounded.
  This makes it easy to set up a multivariate normal prior.
  Working in :math:`(q, h, D)`-space we would need different probability distributions for the parameters, to impose the constraints :math:`0 < h < 1` and :math:`D > 0`.

* **Parameters appear to be less correlated.**
  We did not investigate this in detail, but it appears that parameters are less correlated in :math:`\boldsymbol{\theta}`-space.
  
.. plot:: plots/plot_dca_contours.py
   :show-source-link: True
   
   This figure shows the contours of the loss function in :math:`\boldsymbol{\theta}`-space for a curve fit.
  

Integrals of the reparameterized Arps curve
"""""""""""""""""""""""""""""""""""""""""""

The equation is simply the logarithm of the Arps curve, with an alternative parametrization.
Its integrals are:

.. math::
   \int \exp \left( \mu(t; \boldsymbol{\theta}) \right) \, dt &= - \exp \left( \theta_{1} - e^{- \theta_{3}} \log{\left(t e^{- \theta_{2} + \theta_{3}} + 1 \right)} \right) + C \\
   \int_0^T \exp \left( \mu(t; \boldsymbol{\theta}) \right) \, dt &= e^{\theta_1} - \exp \left( \theta_{1} - e^{- \theta_{3}} \log{\left(T e^{- \theta_{2} + \theta_{3}} + 1 \right)} \right) \\
   \int_T^{\infty} \exp \left( \mu(t; \boldsymbol{\theta}) \right) \, dt &= \exp \left( \theta_{1} - e^{- \theta_{3}} \log{\left(T e^{- \theta_{2} + \theta_{3}} + 1 \right)} \right) \\
   \int_0^{\infty} \exp \left( \mu(t; \boldsymbol{\theta}) \right) \, dt &= \exp \left( \theta_{1} \right)


The mathematical model
----------------------

This section explains how raw input data should be transformed to a tuple :math:`(t, y, w)` for use in a curve fitting routine.
Curve fitting routines minimize an error, and this section also explains how to set up an appropriate error model.

The curve :math:`f(t; \boldsymbol{\theta})` must be tied to the data somehow.
Here is an example of the data that we receive::

   production = [ 10,   8,   5,   3]
   time_on    = [0.9, 0.7, 0.8, 1.0]
   time =       ["2020-01", "2020-02", "2020-03", "2020-04"]

The variable ``production`` contains the total production within each time period.
The variable ``time_on`` is a fraction that indicates the uptime of the well within each period.
The variable ``time`` is mostly used for bookkeeping; it represents the period at each index and can be in yearly, monthly, daily or even hourly resolution.
The typical case is that a period represents a month or a day.

Switching to mathematical notation, let :math:`\boldsymbol{p}` be a vector with observed production in each time period and :math:`\boldsymbol{\tau}` be "time on" (or uptime) within each period.
Let :math:`\xi_i` be the cumulative sum of ``time_on`` up until time period :math:`i`, i.e., :math:`\xi_i = \sum_{j=0}^i \tau_i`.


.. plot:: plots/plot_mathematical_model.py
   :show-source-link: True

   The mathematical model visualized. At the end of each period we observe the total production :math:`p_i` within that period.
   We assume that :math:`p_i` is the integral of an unknown curve :math:`f(t; \boldsymbol{\theta})`.


We assume that production within each time period is the integral of the curve within that period.
Recall that the units for the curve is production per time.
Assuming that each time period represents a small unit of time, we can use the *midpoint rule* :math:`\int_a^b f(t) dt \approx (b - a)f((a+b)/2)` to approximate the integral.
Since decline curves typically are convex and decreasing, the midpoint rule will underestimate slightly.

.. math::
   p_i &= \int_{\xi_{i-1}}^{\xi_i} f(t; \boldsymbol{\theta}) \, dt \approx 
   \left( \xi_i -  \xi_{i-1} \right) f\left( (\xi_i +  \xi_{i-1})/2 ; \boldsymbol{\theta} \right) \\
   &= \tau_i  f\left( (\xi_i +  \xi_{i-1})/2 ; \boldsymbol{\theta} \right) = \tau_i f\left( t_i ; \boldsymbol{\theta} \right)

In the equation above, we define :math:`t_i = (\xi_i +  \xi_{i-1})/2 = \xi_i - \tau_i/2`.
When we use the midpoint rule, we can fit :math:`f(t; \boldsymbol{\theta})` to the observed values directly instead of fitting the integral.
This is simpler, because computing the gradient :math:`\nabla_\boldsymbol{\theta} f(t; \boldsymbol{\theta})` for BFGS is easier than computing the gradient of the integral.

.. plot:: plots/plot_curve_shift.py
   :show-source-link: True

   In this figure we assume the well is always on. 
   The midpoint rule shifts the curve to the left so the integral closely matches the sum.
   If the curve is linear, then the integrals match the sum perfectly.
   However, since decline curves are decreasing and convex, we do end up with a slight bias: the midpoint rule underestimates the integral.

**The error model.**
We will now introduce the error model.
We assume that errors :math:`\epsilon_i \sim N(0, \sigma)` are tied to the order of magnitude of the production rate.
For instance, if the production rate is :math:`100`, then the probability of producing :math:`100 (1.2)` is equal to the probability of producing :math:`100 (1/1.2)`.
Or, if the production rate is :math:`10`, then the probability of producing :math:`10 (1.5)` is equal to the probability of producing :math:`10 (1/1.5)`.
This leads us to fit on the log scale.
Furthermore, we assume that the sum of the errors accumulates over the time duration :math:`\tau_i`, so the summed error becomes :math:`\epsilon_i \sim N(0, \sigma \sqrt{\tau_i})`.
The full model therefore becomes:

.. math::
   \log (p_i) &= \log \left( \tau_i \right) +  \log\left( f\left( t_i; \boldsymbol{\theta} \right) \right) + \epsilon_i \\
   \log \left( \frac{p_i}{\tau_i} \right) &=  \log\left( f\left( t_i ; \boldsymbol{\theta} \right) \right) + \epsilon_i \\
   \log \left( y_i \right) &=  \log\left( f\left( t_i ; \boldsymbol{\theta} \right) \right) + \epsilon_i

In the equation above, we defined the average production rate as :math:`y_i = p_i / \tau_i = p_i / (\xi_i -  \xi_{i-1})`.
Going back to the original scale by exponentiating both sides of the equation above, the model becomes

.. math::
    y_i =  f\left( t_i ; \boldsymbol{\theta} \right) \exp(\epsilon_i) = f\left( t_i ; \boldsymbol{\theta} \right) \exp(N(0, \sigma \sqrt{\tau_i})),

and this tells us how to simulate data from the model, which we'll use to forecast.

Going back to the example data, we can create the following values and pass them to a curve fitting routine::

   production = [ 10,   8,   5,   3]
   time_on    = [0.9, 0.7, 0.8, 1.0]
   time =       ["2020-01", "2020-02", "2020-03", "2020-04"]

   t = [ 0.45,  1.25,    2, 2.9]
   y = [11.11, 11.43, 6.25,   3]
   w = [  0.9,   0.7,  0.8, 1.0]

The weights :math:`w_i` for the regression are obtained using `inverse-variance weighting <https://en.wikipedia.org/wiki/Inverse-variance_weighting>`_.
We're fitting average production rates to the model, so in a time period the variance is :math:`\sigma_i^2 = \sigma^2 / \tau_i`.
Notice the correction for :math:`\tau_i`; the longer the time period is, the lower the variance in the mean is.
Using inverse variance weighting, the weights become 

.. math::
   w_i = \frac{1}{\sigma_i^2} = \frac{\tau_i}{\sigma},

where :math:`\sigma` is the (unknown) uncertainty over one time period will full production.
Finding the best parameters for :math:`\boldsymbol{\theta}` in the loss function is independent of the value of :math:`\sigma` - the optimization problem is separable.
The exact same weighting scheme can alternatively be obtained with a simple argument: data from ten short periods should give the same information as data from one long period that is ten times longer.
Therefore the weighting should be proportional to the duration of each period.

Alternatives to fitting production rates on log-scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Fitting on the original scale.**
Instead of assuming :math:`\log y_i \approx \log f(t_i; \boldsymbol{\theta})`, we could assume that :math:`y_i \approx f(t_i; \boldsymbol{\theta})`.
From a curve fitting perspective, fitting data to model instead of log-data to log-model means that the model will work very hard to minimize errors when :math:`y_i` is large.
Well production data has :math:`y_i`-values on vastly different scales over the lifetime of a well, and large residuals will occur when :math:`y_i` is large.
From a forecasting perspective this is unfortunate, since the most recent history has low production values and matters the most for forecasting (the recent past is more important than the distant past).

From a statistical perspective, the model 

.. math::  
   p_i &= \int_{\xi_{i-1}}^{\xi_i} f(t; \boldsymbol{\theta}) + \epsilon_i \, dt  \approx \tau_i \left[ f\left( t_i ; \boldsymbol{\theta} \right) + \epsilon_i \right] \\
   \frac{p_i}{\tau_i} &= y_i = f\left( t_i ; \boldsymbol{\theta} \right) + \epsilon_i

is nonsensical, since:

1. Errors :math:`\epsilon_i` are independent of the :math:`y_i` values, but this does not reflect real data. If production is on the order of a million units, then we expect completely different errors compared to when the production is on the order of a thousand units.
2. Simulating from this model will lead to negative production values, especially when :math:`t` is large. Then :math:`f\left( t ; \boldsymbol{\theta} \right) \approx 0` and simulating from the error term :math:`\epsilon_i` will produce negative values.

**Fitting cumulatives.**
Again, from a curve fitting perspective we could fit

.. math::  
   \sum_{i=0}^j p_i = \int_{0}^{\tau_j} f(t; \boldsymbol{\theta}) \, dt

for every period :math:`j`, but the issue is again the relative sizes of the residuals as well as the error model.
Where would one place residuals in such a model?

If we add the error term :math:`\epsilon_i` on the original scale, then we must at the very least account for the fact that errors accumulate over the whole history:

.. math::  
   \sum_{i=0}^j p_i = \int_{0}^{\tau_j} f(t; \boldsymbol{\theta}) \, dt + \xi_j \epsilon_i

But the above model also assumes that to errors are constant over time, whereas in reality the errors are clearly dependent on the production values.
A stochastic integral like :math:`\sum_{i=0}^j p_i = \int_{0}^{\tau_j} f(t; \boldsymbol{\theta}) \exp(\epsilon(t)) \, dt` might work, but it does not appear that such a model will lead to any improvements.
It's hard to compute and unnecessarily complex with no apparent gain.

Time series preprocessing
-------------------------

This section outlines two common ways to preprocess data: **the producing time transform** and **the calendar time transform**.

.. plot:: plots/plot_preprocessing.py
   :show-source-link: True
   
   In this figure, the true model is an exponential function.
   At the end of each time period, we observe the total production (dots) that occurred within each time period.
   This production is the integral of the true model over the time on (uptime).
   Two preprocessing transformations are shown: (1) the producing time transform and (2) the calendar time transform.
   Forecasting with (1) answers the question "how much will this well produce in the future if it's always on?" and
   forecasting with (2) answers the question "how much will this well produce in the future if it's future uptime is roughly equal to it's past uptime?"

As in the section above, let :math:`\boldsymbol{p}` be observed production within each time period and
:math:`\boldsymbol{\tau}` be time on.
The variable :math:`\xi_i` denotes the cumulative sum of time on up until time period :math:`i`, i.e., :math:`\xi_i = \sum_{j=0}^i \tau_i`.

**The producing time transform.**
We can use DCA to forecast the future well production potential, i.e., how much a well will produce in the future if it is always producing (no downtime).
For curve fitting, we need a triple :math:`(\boldsymbol{t}, \boldsymbol{y}, \boldsymbol{w})` of :math:`t`-values, :math:`y`-values and weights :math:`w`.
The producing time transform is given by

.. math::
   t_i &= \xi_i - \tau_i/2 \\
   y_i &= p_i / \tau_i \\
   w_i &= \tau_i

To create the :math:`t`-values, we contract time as if the well was always on and place each :math:`t`-value at the midpoint.
To create the :math:`y`-values, we divide production by time on to get production rates.
To create the weights :math:`w`, we use the time on within each period.

**The calendar time transform.**
We can use DCA to forecast how much a well will produce if time on in the future is roughly equal to time on in the past.
The calendar time transform simply sets :math:`\tau_i = 1` for all periods :math:`i`.
The resulting transform is

.. math::
   t_i &= \xi_i - \tau_i/2 = i + 0.5 \\
   y_i &= p_i \\
   w_i &= 1

Which transform to use depends on the question that DCA is supposed to answer.
The producing time transform answers the question "what will this well produce if it's always on in the future?".
The calendar time transform answers the question "what will this well produce if the uptime in the future is roughly equal to the uptime in the past?".
   
Regardless of which transform we choose, some prior data cleaning should be done.
Missing values, zero values and inconsistent values must be dealt with.

Loss functions for DCA
----------------------

The curve fitting problem is, given vectors :math:`(\boldsymbol{t}, \boldsymbol{y}, \boldsymbol{w})`, to find parameters :math:`\boldsymbol{\theta}` such that :math:`f(\boldsymbol{t} ; \boldsymbol{\theta})` is close to :math:`\boldsymbol{y}`.
The function :math:`f` can be the Arps curve, the exponential function, the scaled Weibull distribution, or some other function.
We will assume that :math:`f` is the Arps curve.
See [BOYD04]_ for more on loss and optimization.

Least squares loss
^^^^^^^^^^^^^^^^^^

To our knowledge, all previous DCA in Equinor that used Python have relied upon the optimization routine `scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_.
This routine determines :math:`\boldsymbol{\theta}` as the solution to the least squares optimization problem

.. math::
   \underset{\boldsymbol{\theta}}{\operatorname{minimize}}
   \sum_i^n \left( y_i - f(t_i ; \boldsymbol{\theta}) \right)^2,

which is equivalent to assuming :math:`y_i = f(t_i ; \boldsymbol{\theta}) + \epsilon` with :math:`\epsilon \sim \text{Normal}(0, \sigma)` and maximizing the log-likelihood

.. math::
   \log 	\mathcal{L}(\boldsymbol{\theta} ) & = \log \prod_{i}^{n} \text{Normal}(f(t_i ; \boldsymbol{\theta}), \sigma) \\
   & = \sum_i^n \log  \frac{1}{\sqrt{2 \pi } \sigma} \exp \left( - \frac{(y_i -f(t_i ; \boldsymbol{\theta}) )^2}{2 \sigma^2 } \right) \\
   & \propto \sum_i^n  (y_i -f(t_i ; \boldsymbol{\theta}) )^2

with respect to :math:`\boldsymbol{\theta}`.

The loss function defined by the likelihood above is sensible if the assumptions hold.
Here, the assumptions are roughly speaking that:

* The errors :math:`\epsilon` are normally distributed. If they are not, then the model will be very sensitive to outliers.
  In least squares, doubling the error increases the penalty by four.

* We care about errors on the original scale. If we care more about relative errors, or errors on the multiplicative scale, then least squares is not appropriate.

* We care equally about each data point. If we care more about fitting to recent data (it's sensible to assume that the most recent past is more predictive of the future than the distant past), then least squares is not appropriate.
  Fitting all previous data well does not mean that we predict future data as well as we could.
  Weighting the most recent data more heavily might be more sensible in forecasting problems.

Our position is that in DCA, none of these assumptions hold up very well.
We suggest customizing the loss function instead of using the default ``scipy.optimize.curve_fit`` routine.
A custom loss function can be used in conjunction with ``scipy.optimize.minimize`` to fit curves.

An improved loss function
^^^^^^^^^^^^^^^^^^^^^^^^^

We propose the loss function

.. math::
   \mathcal{L}(\boldsymbol{\theta}) = \sum_i w_i W(t_i, \gamma) |\log y_i - \log f(t_i; \boldsymbol{\theta}) |^p + \alpha \left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right)^T \boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1} \left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right),

where :math:`w_i`, :math:`y_i` and :math:`t_i` are data and :math:`\boldsymbol{\theta}` is the unknown vector of curve parameters that we're optimizing.
This equation is implemented as :py:class:`CurveLoss`.

The least-squares loss is modified by adding the following terms:

* :math:`W(t_i, \gamma)` is a weighting function that scales up the contribution of the most recent data,
* the weighting function is :math:`W(t_i, \gamma)` is parametrized by the *half-life* :math:`\gamma`,
* :math:`1 \leq p \leq 2` is a hyperparameter determined by cross-validation,
* a multivariate normal prior is put on :math:`\boldsymbol{\theta}` (the second term with the quadratic function),
* the strength of the prior is given by :math:`\alpha` and determined by cross-validation.

Notice how :math:`\mathcal{L}` is simply an extension of the least squares loss.
We allow :math:`p` to vary between :math:`2` (MSE, minimized by the arithmetic mean) and :math:`1` (MAE, minimized by the median).
Then we add weights to more recent data points (since the most recent past might be more predictive of the future if the task is forecasting).
Finally we add a multivariate normal prior by introducing a prior mean :math:`\boldsymbol{\mu}_{\boldsymbol{\theta}}` and prior covariance :math:`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}`.
The least squares loss is a subset of this loss, and if least squares leads to the best performance, then cross-validation will tell us so.

**Recommendation.** 
Choose a cross validation strategy that mimics, as closely as possible, how the model will be used.
If the goal is to predict the future, then we recommend splitting each time series in half (or at a random point in time), then training on the first half, and optimizing for hyperparameters that best help predict the second half of the time series.
If the goal is to predict how much the well will produce over it's lifetime, then split in half, train on the first half, and optimize for hyperparameters that help the curve fit both the first and second half.

We will now discuss the loss function in more detail.

Multiplicative errors
"""""""""""""""""""""

It might be better to use multiplicative errors compared to squared errors, i.e. fitting the log-model against the log data.
Consider the simplest model: fitting a mean value by :math:`f(\boldsymbol{t}, \theta) = \theta`.
Fitting to data :math:`\boldsymbol{y} = \left(1, 10, 100\right)` by minimizing least squares gives the arithmetic mean :math:`\theta = 37`.
Loss above and below balance each other perfectly on the additive scale.

Fitting to the same data by minimizing log model against log data gives :math:`\log_{10}(\theta) = 1`, so :math:`\theta = 10`. Loss above and below are balanced in the multiplicative sense, since multiplying by :math:`10` is as erroneous as diving by :math:`10`.

Fitting without taking logs forces the optimizer to work hard at fitting the first few data points at the expensive of the last data points.
This is due to the magnitude of the errors in the beginning of the time series, when production is large due to the high pressure in the well.
When we are trying to predict the future, focusing on the first errors at the expense of the last errors in time series seems counterproductive.

.. plot:: plots/plot_log_vs_nolog.py
   :show-source-link: True
   
   Two curves fitted to data.
   Fitting to :math:`y = a + bt` in log-space and exponentiating the result is not equivalent to fitting :math:`y = \exp (a + bt)` in normal-space.
   

Setting :math:`p` to be a hyperparameter
""""""""""""""""""""""""""""""""""""""""

We've observed that when we try to minimize forecasting error as measured by RMSE, the exponent :math:`p` in :math:`\mathcal{L}` should often be closer to :math:`1`, rather than :math:`2`.
This is a bit surprising, since choosing :math:`p=2` means choosing the loss function that minimizes the RMSE metric.
However, when :math:`p=1` we're fitting the median curve (percentile regression), which is more robust to outliers.
We propose to let :math:`p \in [1, 2]` be a hyperparameter, to be determined by cross validation on the data and prediction task.
   
.. plot:: plots/plot_rmse_vs_mae.py
   :show-source-link: True
   
   Two curves fitted to data with outliers.
   The curves use different :math:`p` norms, and we observe that :math:`p=1` is robust to outliers.

Weight recent data points more heavily
""""""""""""""""""""""""""""""""""""""

We propose adding weights :math:`W(t_i, \gamma)` to each term in the loss function.
The loss function then becomes

.. math::
   \sum_i^n W(t_i, \gamma) w_i  |\log y_i - \log f(t_i; \boldsymbol{\theta}) |^p

We set :math:`W(t_i, \gamma) = C 2^{(t_i - \max_i t_i) / \gamma}`, where :math:`\gamma` is the half-life of the exponential decay and :math:`C` is a constant.
For instance, a half-life of :math:`\gamma = 12` months means that the 12th most recent data point gets a relative weight of :math:`50 \%`, the 24th data point gets a weight of :math:`25 \%`, the 36th data point gets a weight of :math:`12.5 \%`, and so forth.
We determine :math:`C` by the constraint :math:`\sum_{i=1}^n W(t_i, \gamma) = n`.
The half life :math:`\gamma > 0` becomes a hyperparameter, to be determined by cross validation on the data and prediction task.

.. plot:: plots/plot_weighting_by_recency.py
   :show-source-link: True
   
   A small half-life leads the curve to fit more recent data better.


Use priors on the parameters
""""""""""""""""""""""""""""

Finally we set a prior on :math:`\boldsymbol{\theta}`.
A multivariate normal prior leads to a term with a quadratic form :math:`\left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right)^T \boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1}  \left( \boldsymbol{\theta} - \boldsymbol{\mu}_{\boldsymbol{\theta}}  \right)` in the log-likelihood.
The prior is chosen based on inspecting the data set, and it serves several purposes, since it:

* makes fitting a curve when we have fewer data points than parameters possible,
* helps regularize the curve when there is little information,
* makes the numerical routines more stable, and provides an alternative to hard constraints on the parameter space.

In :math:`\mathcal{L}` we set a parameter :math:`\alpha` in front of the prior, which determines regularization strength.
This parameter could alternatively be absorbed into :math:`\boldsymbol{\Sigma}_{\boldsymbol{\theta}}^{-1}`, but separating it out can be helpful.

For more on Bayesian statistics, see [BISH11]_, [MCEL20]_, [GELM13]_

Uncertainty and EUR-forecasting
-------------------------------

This section contains information about how to incorporate uncertainty into forecasts with Arps curves.

Note that:

* We do not consider full posterior predictive uncertainty, since we ignore uncertainty in :math:`\boldsymbol{\theta}`. 
  This uncertainty could be incorporated, but would make the code more complex and potentially more slow.
  A `Laplace approximation <https://en.wikipedia.org/wiki/Laplace%27s_approximation>`_ or full `Markov chain Monte Carlo <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ could be used to capture this uncertainty.
  But we believe that it is unnecessary; we typically have a lot of data (wells on steady decline require significant history) and in that case the uncertainty in the parameters :math:`\boldsymbol{\theta}` is low.
* The uncertainty that we capture is only with respect to the model and the data, not with respect to everything that might happen to a real-life well (interventions, stopping, adding other wells, injectors, etc).
* We take a pragmatic approach: the goal is to get good probability estimates in reasonable time, not construct a perfect theoretical model.

On the original scale
^^^^^^^^^^^^^^^^^^^^^

If we assume normally distributed errors on the observed scale (not log-scale), we can construct a probability model like

.. math::
   y_i \sim \mathcal{N}\left( f(t_i ; \boldsymbol{\theta}), \sigma \right),
   
which is equivalent to the model :math:`y_i = f(t_i ; \boldsymbol{\theta}) + \epsilon_i` with :math:`\epsilon_i \sim \mathcal{N}(0, \sigma)`.
This is not a very realistic model: we can estimate negative production, and the model assumes that the errors are on the absolute scale.
Still it is instructive to look at this model first, before we study a more realistic log-scale model.

Suppose we wish to estimate the EUR (Estimated Ultimate Recovery) with associated uncertainty.
We have observed :math:`n` data points so far, and the final time we have observed is :math:`t_n`.
We must first form the sum :math:`\sum_{i=0}^n y_i` of what we have observed,
then we must sum our predictions for the future :math:`\sum_{i=n+1}^N \hat{y}_i`.

The EUR calculation with uncertainty becomes:

.. math::
   \text{EUR} = \sum_{i=0}^n y_i + \sum_{i=n+1}^N \hat{y}_i &= \sum_{i=0}^n y_i + \sum_{i=n+1}^N \left( f(t_i ; \boldsymbol{\theta}) + \epsilon_i \right) \\
    = \sum_{i=0}^n y_i + \sum_{i=n+1}^N f(t_i ; \boldsymbol{\theta}) + \sum_{i=n+1}^N \epsilon_i &= \sum_{i=0}^n y_i + \sum_{i=n+1}^N f(t_i ; \boldsymbol{\theta}) + \mathcal{N}(0, \sigma \sqrt{N - n}) 

where we have used fact that a sum of :math:`n` normal variables :math:`\mathcal{N}(0, \sigma)` is a normal variable with distribution :math:`\mathcal{N}(0, \sigma \sqrt{n})`.
The EUR decomposes into three parts: the sum of observed production in each period where we have data,
the integral of the model from the last observed period forward in time,
and the random noise.

.. math::
   \text{EUR} = \underbrace{\sum_{i=0}^n y_i}_\text{observed} + 
   \underbrace{\int_{t_{n+1}}^{t_{N}} f(t_i ; \boldsymbol{\theta}) dt }_\text{predicted mean} + 
   \underbrace{\mathcal{N}(0, \sigma \sqrt{N - n} )}_\text{noise}
   
Only the last term is stochastic, so the uncertainty is captured in that term, which has a closed-form distribution.
Notice how the uncertainty range grows like :math:`\sqrt{n}` into the future.

On the log-scale
^^^^^^^^^^^^^^^^

The model above with errors on the original scale assumes that the magnitude of the errors are independent of the size of the observations :math:`y_i`.
In reality, a more accurate model is to assume that errors are on the log scale.
For instance, going 20% over the predicted value has a constant probability, no matter what the production value :math:`y_i` is.
To accomplish this we fit the log-model to the log-data, and the model becomes

.. math::
   \log ( y_i ) \sim \mathcal{N}\left( \log f(t_i ; \boldsymbol{\theta}), \sigma \right),
   
which is equivalent to :math:`\log (y_i) = \log \left( f(t_i ; \boldsymbol{\theta}) \right) + \epsilon_i` with :math:`\epsilon_i \sim \mathcal{N}(0, \sigma)`.

To obtain EUR, we again sum the estimates with the uncertainty:

.. math::
   \text{EUR} = \sum_{i=0}^n y_i + \sum_{i=n+1}^N \hat{y}_i &= \sum_{i=0}^n y_i + \sum_{i=n+1}^N \exp( \log \hat{y}_i ) \\
   = \sum_{i=0}^n y_i + \sum_{i=n+1}^N \exp( \log \left( f(t_i ; \boldsymbol{\theta}) \right) + \epsilon_i )
   &= \sum_{i=0}^n y_i + \sum_{i=n+1}^N \exp( \epsilon_i )  f(t_i ; \boldsymbol{\theta})
   
In this model, final prediction term is a sum-of-products, so we cannot decompose the randomness from the predicted mean.
However, the equation above can be Monte-Carlo simulated: by drawing random values of :math:`\epsilon_i` and computing sums we obtain the distribution of the EUR.

The errors in the log-scale model implicitly depend on the time resolution
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The log-scale model is not invariant to re-sampling the data at finer or coarser resolutions.
The model depends on the data resolution, since it places a log-normal error on each period and the sum of lognormals is not lognormal.
This means that using the model on monthly resolution vs. weekly resolution will not produce exactly equal results.

The practical take-away is that the resolution should be such that the model equation :math:`\log ( y_i ) \sim \mathcal{N}\left( \log f(t_i ; \boldsymbol{\theta}), \sigma \right)` is reasonable.
In other words: look at the residuals and see that they are approximately normal in log-space.

We now show the implicit dependence on the resolution.
Let :math:`\tau_i` be time on for an arbitrary period with production :math:`p_i`.
Spelling out the model for some period :math:`i=1`, we have

.. math::
   \log ( p_1 ) = \log \left( \int_{\xi_{i-1}}^{\xi_i} f(t ; \boldsymbol{\theta}) \, dt \right) + \sqrt{ \tau_1 } N(0, \sigma) \approx \log \left( \tau_1 f(t_1 ; \boldsymbol{\theta}) \right) + \sqrt{ \tau_1 } N(0, \sigma).

Now suppose we obtain data on a resolution that is twice as fine, so we have :math:`p_1' + p_2' = p_1`.
This gives us an opportunity to refine the integral approximation.
Still we do have two options for the error model: stay on the original resolution or fit data to the new resolution that is twice as fine.
These two options produce different results, as we will now show.

Assume that :math:`\sigma` is in terms of the new, fine resolution.
If we choose to stay on the original resolution (aggregate-then-fit), the model equation becomes

.. math::
   \log ( p_1 ) = \log ( p_1' + p_2' ) = \log \left( \tau_1' f(t_1' ; \boldsymbol{\theta}) + \tau_2' f(t_2' ; \boldsymbol{\theta}) \right) + \sqrt{ \tau_1' + \tau_2' } N(0, \sigma).

And the prediction for :math:`p_1 = p_1' + p_2'` becomes a number times a log-normal stochastic variable:

.. math::
   p_1 = \left[ \tau_1' f(t_1' ; \boldsymbol{\theta}) + \tau_2' f(t_2' ; \boldsymbol{\theta}) \right]
         \exp \left( \sqrt{ \tau_1' + \tau_2' } N(0, \sigma)  \right)

If we fit to the new, finer resolution (fit-then-aggregate), the model equations become

.. math::
   \log ( p_1' ) &= \log \left( \tau_1' f(t_1' ; \boldsymbol{\theta})  \right) + \sqrt{ \tau_1' } N(0, \sigma) \\
   \log ( p_2' ) &= \log \left( \tau_2' f(t_2' ; \boldsymbol{\theta})  \right) + \sqrt{ \tau_2' } N(0, \sigma).

And the prediction for :math:`p_1 = p_1' + p_2'` becomes the sum of two log-normal stochastic variables:

.. math::
   p_1 = \tau_1' f(t_1' ; \boldsymbol{\theta}) \exp \left( \sqrt{ \tau_1' } N(0, \sigma)  \right)  
   + \tau_2' f(t_2' ; \boldsymbol{\theta})  \exp \left( \sqrt{ \tau_2' } N(0, \sigma)  \right).

With no uncertainty, the result is the same since :math:`\exp\left( \log ( p_1' + p_2' ) \right) = \exp\left( \log p_1' \right) + \exp\left( \log p_2' \right)`.
However, with uncertainty the result is not the same, since:

- In the original coarse resolution, the prediction has a lognormal error.
- In the new finer resolution, the prediction has an error that is the sum of two lognormals.

In theory this means that the error model is tied to the data resolution, and our log-scale model is not independent of data resolution.
In practice this means that we should ensure that the model assumptions are valid.
This is not hard: plot a few wells and see if errors are approximately normal on the log-scale for a given resolution.
If it is not, then consider using a different resolution if possible.
At the end of the day the most important thing is whether we make good predictions, though it is somewhat displeasing that that the model depends on the data resolution.

A central limit theorem approximation to EUR
""""""""""""""""""""""""""""""""""""""""""""

This section contains equations for a central limit approximation to the EUR.
In practice we recommend Monte-Carlo simulation instead of using this approximation, but it is still of interest to sketch it out.

The term :math:`\sum_{i=n+1}^N \exp( \epsilon_i )  f(t_i ; \boldsymbol{\theta})` above is a weighted sum of exponentiated normal variables.
The weights are :math:`f(t_i ; \boldsymbol{\theta})` and the randomness is in the :math:`\exp( \epsilon_i )` terms.
If the weights decay quickly, then the first terms will dominate and the sum will not be normally distributed.
However, if the weights decay slowly and there are many terms in the sum, then we can appeal to the `central limit theorem <https://en.wikipedia.org/wiki/Central_limit_theorem>`_ and approximate the distribution of the EUR.

Using the `properties of the log-normal distribution <https://en.wikipedia.org/wiki/Central_limit_theorem>`_, the linearity of expectations and the `variance of a sum <https://en.wikipedia.org/wiki/Variance>`_, we obtain the following equations:

.. math::
   \mathbb{E} \left[ w e^{\epsilon_i} \right] &= \mathbb{E} \left[ w e^{ \mathcal{N}(0, \sigma)} \right] = w e^{\sigma^2/2} \\
   \mathbb{V} \left[ w e^{\epsilon_i} \right] &= \mathbb{V} \left[ w e^{ \mathcal{N}(0, \sigma)} \right] = w^2 \left( e^{\sigma^2} - 1 \right) e^{\sigma^2}
   
Using these relationships on the stochastic term :math:`\sum_{i=n+1}^N \exp( \epsilon_i )  f(t_i ; \boldsymbol{\theta})` above, we can obtain its expected value:

.. math::
   \mathbb{E} \left[ \sum_i \exp( \epsilon_i )  f(t_i ; \boldsymbol{\theta}) \right] &= 
   \sum_i \mathbb{E} \left[ \exp( \epsilon_i ) \right] f(t_i ; \boldsymbol{\theta}) \\
   = \sum_i \left( \exp \left( \sigma^2/2 \right) \right)  f(t_i ; \boldsymbol{\theta})  &=
   \exp \left( \sigma^2/2 \right) \sum_i  f(t_i ; \boldsymbol{\theta})  \\
   &\approx \exp \left( \sigma^2/2 \right) \int  f(t ; \boldsymbol{\theta}) \, dt.
   
The computation for the variance is similar:

.. math::
   \mathbb{V} \left[ \sum_i \exp( \epsilon_i )  f(t_i ; \boldsymbol{\theta}) \right] &= 
   \sum_i \mathbb{V} \left[ \exp( \epsilon_i ) \right] f(t_i ; \boldsymbol{\theta})^2 \\
   = \sum_i \left( \left(\exp(\sigma^2) - 1 \right) \exp(\sigma^2) \right) f(t_i ; \boldsymbol{\theta})^2  &=
   \left(\exp(\sigma^2) - 1 \right) \exp(\sigma^2)  \sum_i  f(t_i ; \boldsymbol{\theta})^2 \\
   &\approx  \left(\exp(\sigma^2) - 1 \right) \exp(\sigma^2)  \int f(t ; \boldsymbol{\theta})^2 \, dt.
   
   
The normal approximation to EUR, valid only for large predictive horizons with slow decay, becomes

.. math::
   \text{EUR} &\approx \sum_{i=0}^n y_i + \mathcal{N}\left( \mu_{\text{EUR}}, \sigma_{\text{EUR}} \right) \\
   \mu_{\text{EUR}} =& \exp \left( \sigma^2/2 \right) \int_{t_n}^T  f(t ; \boldsymbol{\theta}) \, dt \\
   \sigma_{\text{EUR}}^2 =& \left(\exp(\sigma^2) - 1 \right) \exp(\sigma^2)  \int_{t_n}^T f(t ; \boldsymbol{\theta})^2 \, dt.

Estimating the standard deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In curve fitting we optimize for :math:`\boldsymbol{\theta}`, but to compute the uncertainty we must also optimize for :math:`\sigma`.
There are two ways to do this: (1) in a joint optimization routine where :math:`(\boldsymbol{\theta}, \sigma)` are simultaneously estimated or (2) using the fact that the optimization problem is separable to first estimate :math:`\boldsymbol{\theta}`, then estimate :math:`\sigma` from the residuals.
We take the latter approach, since it affords us more flexibility and can help make the estimation of :math:`\sigma` more robust.

Since the modified loss function uses exponentially decaying weights to estimate curve parameters :math:`\boldsymbol{\theta}`, the same approach should be taken when estimating the standard deviation :math:`\sigma`.
Here too we do not believe that old data is as representative as new data.

There are at least three approaches to estimating uncertainty:

- We can use the **weighted standard deviation** of the residuals, with the same weights that were used to fit the curve.
  However, while the weights are low for old data, the squared residuals are still very large.
  This can lead to over-estimated standard deviation.
- A more robust approach is to use the **Median Absolute Deviation** (MAD), see `the Wikipedia entry on robust measure of scale <https://en.wikipedia.org/wiki/Robust_measures_of_scale>`_.
  Again we should weight the data when computing the MAD, using the same weights that we used when estimating curve parameters.
- A third option is not estimate :math:`\sigma` at all, but instead use a **weighted bootstrap** of the residuals to capture future uncertainty.
  However, this approach does not let us create P10 and P90 curves.

We believe the MAD approach is the best one, since it is robust.

P10 and P90 curves
""""""""""""""""""

Another advantage of fitting on the log-scale is that P10 and P90 curves are easy and natural to construct.
Under our :math:`\boldsymbol{\theta}` parametrization of the Arps cruve we have

.. math::
   \log \left( f(t ; \theta_1, \theta_2, \theta_3) \right) + \alpha \sigma = f(t ; \theta_1 + \alpha \sigma, \theta_2, \theta_3),
   
so constructing curves for percentiles like P10 and P90 is easily done.
This amounts to multiplying :math:`q` with :math:`\exp(\alpha \sigma)` in the original parametrization.
To see why this is true, recall the relationship :math:`q = \exp(\theta_1) (1-h ) D` between the two parametrization, so

.. math::
   \exp(\theta_1 + \alpha \sigma) (1-h ) D = \exp(\alpha \sigma) \exp(\theta_1) (1-h ) D = \exp(\alpha \sigma) q.

The same does not apply if we fit in the original space, since :math:`f(t ; \boldsymbol{\theta}) + \alpha \sigma` is not in the family of Arps curves (it does not decay to zero).

Note that P10 and P90 curves represent *pointwise uncertainty for each individual period*, and cannot be integrated directly to obtain the EUR.
The sum of P90 values of each period is not equal to the P90 of the sum of periods, so simulation must be performed as outlined above.
   
   
The AR(1) regression model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Forward simulation
""""""""""""""""""

Real wells exhibit autocorrelation that can sometimes be quite strong.
We capture autocorrelation by the parameter :math:`\phi` using the following model,
which is a blend of an AR(1) process and a non-linear regression.

.. math::
   \log (y_i) &= \log(f(t_i; \theta) ) + \eta_i \\
   \eta_i &= \phi \eta_{i-1} + \sqrt{\tau_i} \epsilon_i \\
   \epsilon_i &\sim \operatorname{GN}(\mu=0, \alpha=\sigma, \beta=\beta)
   
The parameters are :math:`(\theta, \phi, \sigma, \beta)` and :math:`\operatorname{GN}` is the `generalized normal distribution <https://en.wikipedia.org/wiki/Generalized_normal_distribution>`_.
Given data :math:`(\tau_i, t_i)` and parameters :math:`(\theta, \phi, \sigma, \beta)`, 
the model above gives a recipe for how to simulate :math:`y_i`.

.. plot:: plots/plot_ar1_simulation.py
   :show-source-link: True
   
   Simulations from the model, showing the DCA curve in black.

If the purpose was to merely estimate the curve parameters :math:`\theta`, 
then whether we use independent errors or an AR(1) structure matters little.
However, when it comes to predicting uncertainty in forecasts by forward simulation, 
modeling autocorrelation helps a great deal.
Modeling autocorrelation helps create realistic-looking forecasts and therefore realistic uncertainty estimates.

The probability density function of the GN is given by:

.. math::
   p(y \mid \mu, \alpha, \beta) = \frac{\beta}{2\alpha\Gamma(1/\beta)} \exp\left( -(|y-\mu|/\alpha)^\beta \right)

Parameter estimation
""""""""""""""""""""

To infer parameters, we need a posterior distribution to maximize.
Recall that the posterior answers the question:
given observed data :math:`(\tau_i, t_i, y_i)`, what are the most likely parameters?

Let us first consider the initial condition.
At steady state, the `AR1 process <https://en.wikipedia.org/wiki/Autoregressive_model#Example:_An_AR(1)_process>`_ has a steady state variance

.. math::
   \frac{\sigma^2_\epsilon}{1 - \phi^2}
   =
   \frac{\alpha^2 \Gamma(3/\beta)}{\Gamma(1/\beta) (1 - \phi^2)}
   =
   \frac{\sigma^2 \Gamma(3/\beta)}{\Gamma(1/\beta) (1 - \phi^2)},
   
where :math:`\Gamma(\cdot)` is the `gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.
   
If :math:`\phi \approx 0`, then the steady state distribution is a generalized normal distribution.
If :math:`\phi \gg 0`, then the steady state is a weighted sum of generalized normal distributions, 
and we appeal to the central limit theorem and approximate the steady state as normally distributed.
Our approximation is only wrong if :math:`\phi \approx 0` and :math:`\beta \ll 2`, 
and even then it's only wrong on one data point (the initial condition).

The likelihood steady-state initial condition becomes:

.. math::
   \log y_0
   &\sim 
   N \left( \log( f(t_0; \theta) ), \sigma_0 \right )  \\
   \sigma_0 &= \sigma \frac{\sqrt{\Gamma (3 / \beta)}}{\sqrt{\Gamma(1/\beta) (1 - \phi^2)}}
   

The remaining terms have likelihood:

.. math::
   \log (y_i) \mid \log (y_{i-1})
   &\sim 
   GN \left( \mu_i, \sqrt{\tau_i} \sigma, \beta  \right ) \\
   \mu_i 
   &=
    \log( f(t_i; \theta) ) + \phi \left[ \log (y_{i-1}) - \log(f(t_{i-1}; \theta) ) \right]
   
Gradient of log-likelihood - general case
"""""""""""""""""""""""""""""""""""""""""

On an arbitrary data point that is not the first once (where the initial condition applies), 
we obtain the gradient by first looking at the negative log-pdf of the generalized normal distribution:

.. math::
   L( \log(y_i) \mid \mu, \alpha, \beta) = 
   \frac{\lvert \log(y_i) - \mu \rvert ^\beta}{\alpha^\beta }
   + \log(2\alpha\Gamma(1/\beta)) - \log(\beta)
   
Using the `Prais–Winsten trick <https://en.wikipedia.org/wiki/Prais%E2%80%93Winsten_estimation>`_,
we transform the regression with a weighted difference, creating independent errors to regresson on.

.. math::
   \mu &= \mu(\theta, \phi) = \log( f(t_i; \theta) ) + \phi \left[ \log (y_{i-1}) - \log(f(t_{i-1}; \theta) ) \right]  \\ 
   \alpha &= \alpha(\sigma) = \sqrt{\tau_i} \sigma
   
Let us put an arbitrary likelihood weight :math:`w(\beta)` on the negative log-likelihood.
Differentiating with respect to the curve parameters :math:`\theta`, we obtain

.. math::
   \partial_\theta L &= 
   - \frac{\beta \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} w{\left(\beta \right)} \frac{\partial}{\partial \theta} \mu{\left(\theta,\phi \right)}}{\log(y_i) - \mu{\left(\theta,\phi \right)}} \\
   &= 
   - \frac{\beta \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} w{\left(\beta \right)} 
   \left( \partial_\theta \log( f(t_i; \theta) ) - \phi  \partial_\theta \log(f(t_{i-1}; \theta) )  \right)
   }{\log(y_i) - \mu{\left(\theta,\phi \right)}} 
   
as long as we have the gradient of the curve parameters available, i.e. :math:`\partial_\theta \log(f(t; \theta) )`, the above can be computed.

Differentiating with respect to the autocorrelation parameter :math:`\phi`, we obtain

.. math::
   \partial_\phi L &= 
   - \frac{\beta \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} w{\left(\beta \right)} \frac{\partial}{\partial \phi} \mu{\left(\theta,\phi \right)}}{\log(y_i) - \mu{\left(\theta,\phi \right)}} \\
   &= 
   - \frac{\beta \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} w{\left(\beta \right)} 
   \left( \log (y_{i-1}) - \log(f(t_{i-1}; \theta) ) \right)
   }{\log(y_i) - \mu{\left(\theta,\phi \right)}} 

Differentiating with respect to the error parameter :math:`\sigma`, we obtain

.. math::
   \partial_\sigma L &= 
   - \frac{\left(\beta \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} - 1\right) w{\left(\beta \right)} \frac{d}{d \sigma} \alpha{\left(\sigma \right)}}{\alpha{\left(\sigma \right)}} \\
   &=
   - \frac{\left(\beta \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} - 1\right) w{\left(\beta \right)} \sqrt{\tau_i} }{\alpha{\left(\sigma \right)}} \\
   &=
   - w (\beta) \left( \frac{\beta \left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|^\beta}{(\sigma \sqrt{\tau_i})^\beta} - \frac{1}{\sigma} \right)


Finally, differentiating with respect to the scale parameter :math:`\beta`, we obtain

.. math::
   \partial_\beta L &= 
   \left(   \frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} w{\left(\beta \right)} \log{\left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}} \right)} \\
   &+ \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} \frac{d}{d \beta} w{\left(\beta \right)} - \log{\left(\beta \right)} \frac{d}{d \beta} w{\left(\beta \right)} \\
   &+ \log{\left(2 \alpha{\left(\sigma \right)} \Gamma\left(1/\beta\right) \right)} \frac{d}{d \beta} w{\left(\beta \right)} - \frac{w{\left(\beta \right)}}{\beta} - \frac{w{\left(\beta \right)} \psi \left(\frac{1}{\beta} \right)}{\beta^{2}}

where :math:`\psi` is the `digamma function <https://en.wikipedia.org/wiki/Digamma_function>`_.

If :math:`w` is not a function of :math:`\beta`, then the above simplifies to

.. math::
   \partial_\beta L = 
   w \left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}}\right)^{\beta} \log{\left(\frac{\left|{\log(y_i) - \mu{\left(\theta,\phi \right)}}\right|}{\alpha{\left(\sigma \right)}} \right)} - \frac{w}{\beta} - \frac{w \psi \left(\frac{1}{\beta} \right)}{\beta^{2}}

Gradient of log-likelihood - initial condition
""""""""""""""""""""""""""""""""""""""""""""""

In a similar fashion, **gradients of the initial condition can be computed**.
Recall that the initial condition is given by the negative log-pdf of a normal distribution:

.. math::
   L(\log(y_0) \mid \mu, \sigma_0) &= 
   \frac{\left(\log(y_0) - \mu \right)^{2}}{2 \sigma_0^2}
   + \frac{1}{2}\log(2 \pi \sigma_0^2) \\
   \mu{\left(\theta \right)} &= \log( f(t_0; \theta) ) \\
   \sigma_0(\sigma, \beta, \phi) &= \sigma \frac{\sqrt{\Gamma (3 / \beta)}}{\sqrt{\Gamma(1/\beta) (1 - \phi^2)}}
   
Let us differentiate :math:`\sigma_0(\sigma, \beta, \phi)` first:

.. math::
   \partial_\phi \sigma_0(\sigma, \beta, \phi)  &= \sigma_0(\sigma, \beta, \phi) \frac{\phi}{(1- \phi^2)^3} + \sigma_0(\sigma, \beta, \phi) \frac{\phi}{(1- \phi^2) \sqrt{(1- \phi^2)}}  \\
   \partial_\sigma \sigma_0(\sigma, \beta, \phi)  &= \frac{\sigma_0(\sigma, \beta, \phi)}{\sigma} \\
   \partial_\beta \sigma_0(\sigma, \beta, \phi) &= \sigma_0(\sigma, \beta, \phi) \frac{\psi (1/\beta) - 3 \psi (3/\beta)}{2 \beta^2}
   
We can now use the chain rule to differentiate the negative log pdf with respect to the parameters.

Differentiating with respect to the curve parameters :math:`\theta`, we obtain

.. math::
   \partial_\theta L 
   &=
   - \frac{\left( \log(y_0) - \mu{\left(\theta \right)} \right) w{\left(\beta \right)} \frac{d}{d \theta} \mu{\left(\theta \right)}}{\sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}} \\
   &= 
   - \frac{\left( \log(y_0) - \mu{\left(\theta \right)}\right) w{\left(\beta \right)} 
   \partial_\theta \log( f(t_0; \theta) ) 
   }{\sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}}
   
Differentiating with respect to the autocorrelation parameter :math:`\phi`, we obtain

.. math::
   \partial_\phi L &=
   - \frac{\left(\left(\log(y_0) - \mu{\left(\theta \right)}\right)^{2} - \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}\right) w{\left(\beta \right)} \frac{\partial}{\partial \phi} \sigma_{0}{\left(\sigma,\beta,\phi \right)}}{\sigma_{0}^{3}{\left(\sigma,\beta,\phi \right)}} \\
   &=
   - \frac{\left(\left(\log(y_0) - \mu{\left(\theta \right)}\right)^{2} - \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}\right) w{\left(\beta \right)} 
   \sigma_0{\left(\sigma,\beta,\phi \right)} \frac{\phi}{(1- \phi^2)^3}
   }{\sigma_{0}^{3}{\left(\sigma,\beta,\phi \right)}} \\
   &=
   - \frac{\left(\left(\log(y_0) - \mu{\left(\theta \right)}\right)^{2} - \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}\right) w{\left(\beta \right)} 
    \phi
   }{\sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)} (1- \phi^2)^3}

Differentiating with respect to the error parameter :math:`\sigma`, we obtain

.. math::
   \partial_\sigma L &= 
   - \frac{\left(\left(\log(y_0) - \mu{\left(\theta \right)}\right)^{2} - \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}\right) w{\left(\beta \right)} \frac{\partial}{\partial \sigma} \sigma_{0}{\left(\sigma,\beta,\phi \right)}}{\sigma_{0}^{3}{\left(\sigma,\beta,\phi \right)}} \\
   &=
   - \frac{\left(\left(\log(y_0) - \mu{\left(\theta \right)}\right)^{2} - \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}\right) w{\left(\beta \right)} }{\sigma \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}} \\
   &=
   - w(\beta)
   \left( \frac{\left(\log(y_0) - \mu{\left(\theta \right)}\right)^{2}}{\sigma \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)} } - \frac{1}{\sigma} \right)
   
Finally, differentiating with respect to the scale parameter :math:`\beta`, we obtain

.. math::
   \partial_\beta L &= 
   \frac{\left(\left(y - \mu{\left(\theta \right)}\right)^{2} + \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)} \log{\left(2 \pi \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)} \right)}\right) \frac{d}{d \beta} w{\left(\beta \right)}}{2 \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}} \\
   &- \frac{\left(\left(y - \mu{\left(\theta \right)}\right)^{2} - \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}\right) w{\left(\beta \right)} \left( \psi(1/ \beta) - 3 \psi(3/\beta) \right)}{2 \beta^2 \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}} 
   

If :math:`w` is not a function of :math:`\beta`, then the above simplifies to only the second term:

.. math::
   \partial_\beta L &= - \frac{\left(\left(y - \mu{\left(\theta \right)}\right)^{2} - \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}\right) w{\left(\beta \right)} \left( \psi(1/ \beta) - 3 \psi(3/\beta) \right)}{2 \beta^2 \sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}}  \\
   &= - w(\beta) 
   \left( \frac{\left(y - \mu{\left(\theta \right)}\right)^{2} }{\sigma_{0}^{2}{\left(\sigma,\beta,\phi \right)}} - 1 \right)
   \frac{ \psi(1/ \beta) - 3 \psi(3/\beta) }{2 \beta^2}

Summary
-------
.. _section:summary:

Even though the Arps curve dates back to the 1940s, we still believe it's the best choice when predicting future prediction for on-shore gas wells.
While other models might also be worth examining, we find that the largest performance gains are arhieved when adjusting the loss function.
In other words, how recent data is weighted, how data is reprocessed, whether we are sensitive to outliers or not, and other questions like these, matter more than the exact parametric form of the DCA model.

We back our claim up with a quantitative study based on 15,000 on-shore gas wells.
Keeping the DCA model constant (using the Arps curve) and adjusting the loss function, we obtain a 25% reduction in forecasting RMSE.

.. [BISH11] Bishop, Christopher M., "Pattern Recognition and Machine Learning," Springer, New York, April 2011. `ISBN 978-0-387-31073-2 <https://www.springer.com/gp/book/9780387310732>`_.
.. [MCEL20] McElreath, Richard, "Statistical Rethinking: A Bayesian Course with Examples in R and STAN," 2nd Edition, Chapman and Hall/CRC, Boca Raton, March 2020. `ISBN 978-0-367-13991-9 <https://www.routledge.com/Statistical-Rethinking-A-Bayesian-Course-with-Examples-in-R-and-STAN/McElreath/p/book/9780367139919>`_.
.. [LEE21] Lee, Se Yoon and Mallick, Bani, "Bayesian Hierarchical Modeling: Application Towards Production Results in the Eagle Ford Shale of South Texas," Sankhya B, January 2021. `DOI 10.1007/s13571-020-00245-8 <https://doi.org/10.1007/s13571-020-00245-8>`_.
.. [GELM13] Gelman, Andrew, et al., "Bayesian Data Analysis," 3rd Edition, Chapman and Hall/CRC, Boca Raton, November 2013. `ISBN 978-1-4398-4095-5 <https://www.crcpress.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955>`_.
.. [ARP45] Arps, J.J., "Analysis of Decline Curves," Transactions of the AIME, Vol. 160, pp. 228-247, December 1945. `DOI 10.2118/945228-G <https://onepetro.org/TRANS/article/160/01/228/161823/Analysis-of-Decline-Curves>`_.
.. [JOCH96] Jochen, V. A. and Spivey, J. P., "Probabilistic Reserves Estimation Using Decline Curve Analysis with the Bootstrap Method," SPE Annual Technical Conference and Exhibition, October 1996. `DOI 10.2118/36633-MS <https://onepetro.org/SPEATCE/proceedings/96SPE/All-96SPE/SPE-36633-MS/58944>`_.
.. [BOYD04] Boyd, Stephen and Vandenberghe, Lieven, "Convex Optimization," Cambridge University Press, 2004.
.. [TANG21] Tang, Huiying, et al., "A novel decline curve regression procedure for analyzing shale gas production," Journal of Natural Gas Science and Engineering, Vol. 88, 103818, 2021. `DOI 10.1016/j.jngse.2021.103818 <https://doi.org/10.1016/j.jngse.2021.103818>`_.
.. [LIU18] Tan, Lei, et al., "Methods of Decline Curve Analysis for Shale Gas Reservoirs," Energies, Vol. 11, No. 3, 552, 2018. `DOI 10.3390/en11030552 <https://www.mdpi.com/1996-1073/11/3/552>`_.
.. [LIANG20] Liang, Hong-Bin, et al., "Empirical methods of decline-curve analysis for shale gas reservoirs: Review, evaluation, and application," Journal of Natural Gas Science and Engineering, Vol. 83, 103531, 2020. `DOI 10.1016/j.jngse.2020.103531 <https://doi.org/10.1016/j.jngse.2020.103531>`_.