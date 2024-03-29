# About

applied_stats is a Python library that implements various statistical distributions and methods in Python. The continuous_distributions module supports plotting and calculating probabilities for normal, chi-squared, t, and F distributions, and the mle module can calculate various maximum likelihood estimators based on particular distributions. I plan to continue development of this to add new functionality and modules in the future. 

The project was inspired while I took MAT 561 (Probability) and MAT 562 (Mathematical Statistics) at the University of Louisville. The goal of this is to provide simple tools to work with applied statistics, while also aiding my own understanding by applying theoretical techniques learned during class.

Please feel free to submit PRs or Issues if you'd like!

## Installation

Run the following to install the applied-stats:

``` 
$ pip install applied-stats
```

This has been tested on Windows 10 and Ubuntu 21.04; will test on macOS soon. 

## Usage

If you prefer a video walkthrough, I have one posted here: [Walkthrough Video](https://youtu.be/jTxofcAS8IY)

Follow these examples to start plotting and calculating probabilities. Various examples, including the ones below, can also be found in the [Demonstration Jupyter Notebook](https://github.com/WillTirone/applied-stats_examples/blob/main/Demonstration.ipynb)

### Generate some plots and calculate some probabilities: 

```python
>>> from applied_stats import continuous_distributions
>>> a = Norm_rv(0,1)
>>> a.plot_pdf()
>>> a.probability_calc()
```

Note: pypi has some difficulty rendering the below plot images, they can be viewed on GitHub.

![link](https://github.com/WillTirone/applied_stats/blob/main/output_images/N(0%2C1)_plot.png)

```python
>>> q = ChiSq_rv(4,crit_value=7)
>>> q.plot_pdf(cv_probability=True)
>>> q.probability_calc()
```
![link](https://github.com/WillTirone/applied_stats/blob/main/output_images/X-sqr(4).png)

### Calculate a two-tailed Chi-Sq distribution and plot: 

```python 
>>> z = st.ChiSq_rv(3)
>>> z.plot_pdf(left_cv=1, right_cv=7, cv_probability=True, two_tail=True)
>>> z.probability_calc()
```
![link](https://github.com/WillTirone/applied_stats/blob/main/output_images/two_tailed_chi_sq_full.PNG)

### Calculate the numeric MLE of several common distributions: 

```python 
>>> from applied_stats import mle 
>>> a = [1,3,2,5,6,7,2,3,4,5]
>>> mle.binomial(10, a)
>>> 0.38

>>> b = [1.2,4.3,2.3,6.8,2.4,3.6]
>>> mle.exponential(b) 
>>> 3.4333333333333336
```

### A basic hypothesis test 

```python 
>>> data = ['R', 'B', 'R', 'B', 'R', 'B', 'R', 'B', 'R', 'B']
>>> a = gen_test(data, 'H0 is that there are 5 red marbles in a bag')
>>> a.run_test(n=4, 
           counter='R', 
           accept_left=3, 
           accept_right=4)
           
>>> 'Do not reject H0. Count is 3'
```

## Tests

Run the tests from the command line with `python test.py -v`
With pytest installed, for more detail, run `pytest -v test.py`
