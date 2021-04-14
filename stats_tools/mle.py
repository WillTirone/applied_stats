import numpy as np 

"""
"The method of maximum likelihood in a sense picks out of all the possible
values of theta the one most likely to have produced the given observations
x1, x2, ..., xn. " (Sahoo, 2008)

"The rationale behind point estimation is quite simple. When sampling is from
a population described by a pdf or pmf f(x|theta), knowledge of theta yields
knowledge of the entire population." (Casella, Berger, 2017)

Every calculation below assumes that each x from the sample is identical and 
independently distributed (iid)

The code below will calculate the numeric value for a given MLE based 
on the distribution; analytical solutions can be found in the text referenced 
below.

References
----------
[1] Sahoo, Prasanna. "Probability and Mathematical Statistics", 
pp 417 (2008).
[2] Casella, G., Berger, R. L., "Statistical Inference"
Belmont (California): Brooks/Cole Cengage Learning pp 337 (2017) 
"""

#continuous distributions: input data can include any real number

def uniform(X): 
    
    """
    if x1,x2,..xn are an iid sample from U(0, theta), the MLE is the 
    nth order statistic, X(n). That is, the largest value of the sample.
    
    References 
    ----------
    [1] Sahoo, "Probability and Mathematical Statistics", pp 420
    [2] Tone, MAT 562: Mathematical Statistics notes, U of L
    """

    return np.max(X)

def exponential(X):
    
    """
    The MLE of the exponential distribution is X-bar, the sample mean.
    
    References
    ----------
    [1] Sahoo, "Probability and Mathematical Statistics", pp 458
    [2] Tone, MAT 562: Mathematical Statistics notes, U of L
    """

    n = len(X)
    
    return np.sum(X) / n

def normal(X): 
    
    """
    if X~N(mu, sigma^2), the MLEs are X-bar and (1/n)*sum(x_i - x-bar)^2 from i
    to n. 
    
    Returns a tuple of the MLE for mu, the mean, and the MLE for variance, the 
    population variance.
    
    (should adjust that formula to be rendered in LaTeX in the docs)
    
    References
    ----------
    [1] Sahoo, "Probability and Mathematical Statistics", pp 422
    [2] Tone, MAT 562: Mathematical Statistics notes, U of L 
    """
    
    X_array = np.array(X) 
    n = len(X_array)
    
    mu_mle = np.sum(X) / n 
    var_mle = (1/n) * np.sum(np.square(X_array - mu_mle))
    
    return mu_mle, var_mle


#discrete distributions: data values MUST countably finite, non-negative ints

#TODO: put discrete value checker here 


def bernoulli(X): 
    
    """
    sample proportion 
    """
    
    #we need to make sure that every data point is an integer
    #if it is not, will throw an exception. This is accomplished mod division
    _input = np.array(X) 
    _int_check = np.equal(np.mod(_input,1),0)
    n = len(_input)  
    
    if np.all(_int_check) == True:
        return np.sum(X) / n 
    else:
        raise ValueError("X must be a discrete data set (only integers)")

def binomial(X):
    
    n = len(X) 
    
    return np.sum(X) / n 

def geometric(): 
    pass 

def poisson():
    pass 
    
    
    
    
    
    
    
    