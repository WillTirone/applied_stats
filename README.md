# Statistics_Module
Creating a project to implement various statistical distributions and methods in Python.

To install and run this project, do the following:

1. Clone the repository to your machine using
```git clone https://github.com/WillTirone/Statistics_Module.git```

2. install the dependencies
```pip install -r requirements.txt```
- alternatively, use the [Anaconda installation](https://www.anaconda.com/) of Python 

3. Follow these examples to start plotting and calculating probabilities. The examples below, and several others, can also be found in the  [Demonstration Jupyter Notebook](https://github.com/WillTirone/Statistics_Module/blob/main/Demonstration.ipynb)

4. To run the test file, from the command line enter: ```python test.py```

```python
>>> from Statistics_OOP import *

>>> a = Norm_rv(0,1)
>>> a.plot_pdf()
>>> a.probability_calc()
```
![link](https://github.com/WillTirone/Statistics_Module/blob/main/output_images/N(0%2C1)_plot.png)

```python
>>> q = ChiSq_rv(4,crit_value=7)
>>> q.plot_pdf(cv_probability=True)
>>> q.probability_calc()
```
![link](https://github.com/WillTirone/Statistics_Module/blob/main/output_images/X-sqr(4).png)
