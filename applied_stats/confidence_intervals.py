import continuous_distributions as st
import mle 

class norm_ci(st.Norm_rv):

    def __init__(self, mean, variance, data, alpha=0.05, estimate='variance'):
        
        """
        Initialize a normal random variable hypothesis test class

        Parameters
        ----------
        
        mean : float / int, mean of normal distribution 
        
        variance : float / int, var of normal distribution 
        
        data : array-like, a list of data to draw from. 
        
        estimate : str, mean or variance, default = mean
        
        Returns
        ---------
        None
        
        Notes
        ---------
        subclass of continuous_distributions.Norm_rv
        
        """

        # need to figure out a way to initialize without needing
        # either a mean or a variance
        
        super().__init__(mean,variance)
        if estimate == 'variance': 
            self.x_bar = mle.normal(data)[0]
            del self.mean 
        elif estimate == 'mean': 
            self.var_hat = mle.normal(data)[1]
            del self.variance
        else:
            raise ValueError('enter either mean or variance for estimate')

        self.alpha = alpha
        
        #lower lim = L = L(x1,...xn)
        #upper lim = U = U(x1,...,xn) 
        #such that P(L <= theta <= U) = 1 - alpha 
        #where theta is the param being estimated 
    