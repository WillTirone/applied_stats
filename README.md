# Statistics_Module

A python library that implements various statistical distributions and methods in Python.

## Installation

Run the following to install the applied-stats:

``` 
pip install applied-stats
```

## Usage

Follow these examples to start plotting and calculating probabilities. Various examples, including the ones below, can also be found in the [Demonstration Jupyter Notebook](https://github.com/WillTirone/applied-stats_examples/blob/main/Demonstration.ipynb)

### Generate some plots and calculate some probabilities: 

```python
>>> from applied_stats import continuous_distributions
>>> a = Norm_rv(0,1)
>>> a.plot_pdf()
>>> a.probability_calc()
```
![link](https://github.com/WillTirone/applied_stats/blob/main/output_images/N(0%2C1)_plot.png)

```python
>>> q = ChiSq_rv(4,crit_value=7)
>>> q.plot_pdf(cv_probability=True)
>>> q.probability_calc()
```
![link](https://github.com/WillTirone/applied_stats/blob/main/output_images/X-sqr(4).png)

### Calculate the numeric MLE of several common distributions: 

```python 
>>> from stats_tools import mle 
>>> a = [1,3,2,5,6,7,2,3,4,5]
>>> mle.binomial(a)
>>> 3.8

>>> b = [1.2,4.3,2.3,6.8,2.4,3.6]
>>> mle.exponential(b) 
>>> 3.4333333333333336
```

## Tests

Run the tests from the command line with `python test.py`
