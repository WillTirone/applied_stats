import numpy as np 

"""

"The method of maximum likelihood in a sense picks out of all the possible
values of theta the one most likely to have produced the given observations
x1, x2, ..., xn. "

Every calculation below assumes that each x from the sample is identical and 
independently distributed (iid). 

The code below will calculate the numeric value for a given MLE based 
on the distribution; analytical solutions can be found in the text referenced 
below.

References
----------
[1] Sahoo, Prasanna. "Probability and Mathematical Statistics",
pp 166 - 169 (2008).

"""

#continuous distributions 

def uniform(X): 
    
    """
    
    if x1,x2,..xn are an iid sample from U(0, theta), the MLE is the 
    nth order statistic, X(n). That is, the largest value of the sample.
    
    References 
    ----------
    [1] Tone, Cristina Oct. 2020, MAT 562: Mathematical Statistics, 
    lecture notes, University of Louisville 
    """

    return np.max(X)

def exponential(n, X):
    
    """
    
    The MLE of the exponential distribution is X-bar, the sample mean.
    
    References
    ----------
    [1] Tone, MAT 562: Mathematical Statistics notes, University of Louisville 
    """
    
    return np.sum(X) / n 

def normal(n, X): 
    
    """
    
    if X~N(mu, sigma^2), the MLEs are X-bar and (1/n)*sum(x_i - x-bar)^2 from i
    to n. 
    
    returns a tuple of the MLE for mu, the mean, and the MLE for variance, the 
    population variance.
    
    (should adjust that formula to be rendered in LaTeX in the docs)
    
    """
    
    X_array = np.array(X) 
    
    mu_mle = np.sum(X) / n 
    var_mle = (1/n) * np.sum(np.square(X_array - mu_mle))
    
    return mu_mle, var_mle


#discrete distributions 

