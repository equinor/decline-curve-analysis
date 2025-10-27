FAQ for adca
============

Typical input errors when the fitted curves looks really bad!
-------------------------------------------------------------

* The search range for the halflife hyperparameter is set to silly
  values, like [3, 36] for daily data. The search values are number of
  datapoints. It depends on the data, but [90, 1000] would make more
  sense with daily data and [3, 36] can be fine for monthly data.