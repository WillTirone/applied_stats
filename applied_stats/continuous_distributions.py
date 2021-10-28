import math as m

import numpy as np
from scipy import integrate
from scipy.special import beta
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class Norm_rv:

    """
    Class to initialize a normal random variable with mean mu and variance
    sigma^2

    Norm_rv(mean, variance, crit_value=0.0)

    References
    ----------
    [1] Sahoo, Prasanna. "Probability and Mathematical Statistics",
    pp 166 - 169 (2008).
    """

    def __init__(self, mean, variance, left_cv=0.0,right_cv=1.0):
        self.mean = float(mean)
        self.sigma = float(m.sqrt(variance))
        if variance >0 and variance < np.inf:
            self.variance = variance
        else:
            raise ValueError('Enter a variance between 0 and infinity')
        self.left_cv = float(left_cv)
        self.right_cv = float(right_cv)

        #adding a scale as an int from variance to scale the plot properly and display the graph
        plot_scale = int(self.variance)
        self.x_range = np.linspace(round(-4*plot_scale),
                                   round(4*plot_scale),
                                   round(500*plot_scale))

    def __repr__(self):
        return (f"Normal distribution with mean {self.mean}, variance "
                f"{self.variance}, left critical value {self.left_cv} "
                f"and right critical value {self.right_cv}")

    def pdf(self):

        """probability density function (pdf) of a normal distribution
        with mean mu and variance sigma^2. 
        
        Parameters
        ----------
        Self

        Returns
        ----------
        numpy.ndarray, pdf of a normal distribution evaluated over the range of x
        values for plotting purposes
        
        Notes
        ----------
        To check that it is, in fact, a pdf, the y values must sum to 1.
        """

        return (1/(self.sigma*m.sqrt(2*m.pi)))*m.e**((-1/2)*((self.x_range-self.mean)/self.sigma)**2)

    def plot_pdf(self, cv_probability=False, two_tail=False):

        """
        This function takes a given normal random variable, uses the pdf that
        was previously calculated, and plots it.

        Parameters
        ----------
        two_tail : bool, default False. If true, will allow two tailed plotting
        
        cv_probability : bool, default False. By default, will fill up to the 
        mean. If changed to True along with two_tail, will plot up to the left 
        critical value and right critical value. 

        Returns
        ----------
        None, plt object displayed
        
        Notes
        ----------
        By default, it shades the probability up to the mean. If a critical 
        value is passed and cv_probability=True, it will plot up to the 
        critical value. If two_tail=True is passed, it will shade the tails to
        the left and right of the critical values. The critical value should be
        negative if you want it to shade both sides, as I have not yet 
        implemented a more flexible version of the shading and calculation.
        """

        def _left_fill_helper(fill_to):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_to,
                              where=(self.x_range < fill_to), color='navy', alpha=0.3)

        def _right_fill_helper(fill_to):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_to,
                              where=(self.x_range > fill_to), color='navy', alpha=0.3)

        plt.title(self.__repr__())
        plt.plot(self.x_range, self.pdf(),linestyle='dashed', color='blue',linewidth=3)

        if cv_probability==False:
            _left_fill_helper(self.mean)
        elif (two_tail == True) & (cv_probability==True):
            _left_fill_helper(self.left_cv)
            _right_fill_helper(self.right_cv)

        plt.tight_layout()
        plt.show()

    def probability_calc(self, two_tail=False, cv_probability=True):

        """
        This calculates either the probability to the left or both the right and
        left tails.

        Probability and error estimate attributes are calculated.

        Parameters
        -----------
        two_tail : bool, default False. If two_tail is True, it will simply multiple the
        probability by 2. This is not too flexible yet, but will work for basic cases.

        Returns:
        -----------
        str, f string with probability
        
        Notes
        ----------
        Because the integration function needs a general x variable, and since
        the pdf from Norm_rv.pdf is evaluated over a range of x-values to plot,
        the normal pdf needs to be redefined in this method.This is true for
        all the other distributions as well.
        """

        f = lambda x: ((1/(self.sigma*m.sqrt(2*m.pi)))*
                       m.e**((-1/2)*((x-self.mean)/self.sigma)**2))

        if two_tail == False:
            self.left_probability, self.left_error_est = integrate.quad(f,-np.inf,self.mean)                                              
            return (f"P(X<{self.left_cv}) is {round(self.left_probability,5)}"
                    f"with an error estimate of {round(self.left_error_est,5)}")
        elif (two_tail == True) & (cv_probability==True):
            self.left_probability, self.left_error_est = integrate.quad(f,-np.inf,self.left_cv)  
            self.right_probability, self.right_error_est = integrate.quad(f,self.right_cv, np.inf)  
            self.total_probability = self.left_probability + self.right_probability
            self.total_error = self.left_error_est + self.right_error_est
            return (
            f'P(X < {self.left_cv}) & P(X > {self.right_cv}) is '
            f' {self.total_probability} with an error estimate of '
            f'{round(self.total_error,5)}')

class ChiSq_rv:

    """
    Class for a Chi-squared random variable with k degrees of freedom

    ChiSq_rv(deg_freedom)

    As degrees of freedom increases to infinity, the Chi-squared distribution
    approximates a normal distribution. You may notice that with >171 degrees 
    of freedom, the math.gamma function returns a range error as this is a very
    large number and exceeds the Python-allowed data type limit.

    Parameters
    ----------
    df : degrees of freedom, int

    Returns
    ----------
    Instance of ChiSq_rv class
    
    Notes
    ----------
    As degrees of freedom increases to infinity, the Chi-squared distribution
    approximates a normal distribution. You may notice that with >171 degrees 
    of freedom, the math.gamma function returns a range error as this is a very
    large number and exceeds the Python-allowed data type limit.

    References
    ----------
    [1] Sahoo, Prasanna. "Probability and Mathematical Statistics",
    pp 392 (2008).
    """

    def __init__(self, df):
        if df > 0:
            self.df = int(df)
        else:
            raise ValueError('Degrees of freedom must be > 0')
        self.mean = df
        self.variance = 2*df
        self.x_range = np.linspace(0, 5*self.df, 2000)

        #initializing these variables so we can reassign values in later methods
        # and so we don't need to run plot_pdf() every single time we want a probability
        self.two_tail = False

    def __repr__(self):
        return f"Chi-squared distribution with {self.df} degrees of freedom."

    def pdf(self):

        """
        This is the probability density function (pdf) of a chi squared
        distribution with k degrees of freedom.

        Parameters
        ----------
        Self

        Returns
        ----------
        numpy.ndarray, pdf of a Chi Squared distribution evaluated over the range of x
        values for plotting purposes
        
        Notes
        ----------
        To check that it is, in fact,a pdf, the y values must integrate to 1.
        
        """

        return ((1/(m.gamma(self.df/2)*2**(self.df/2)))*
                self.x_range**((self.df/2)-1)*m.e**(-self.x_range/2))


    def plot_pdf(self, left_cv=0, right_cv=0, cv_probability=False, two_tail=False):

        """
        This function takes a given Chi-squared random variable, uses the pdf
        that was previously calculated, and plots it.

        Parameters
        ----------
        left_cv : float, left critical value used for the left tail, default = 0
        
        right_cv :  float, right critical value used for the right tail, default = 0
        
        cv_probability : bool, an argument to determine whether or not to use critical values. If
        false, simply shades from degrees of freedom to infinity and calculates that probability. If true,
        and two_tail = True, will shade two tails. Default = false.
        
        two_tail : bool, used in combination with cv_probability will plot and shade two-tailed
        chi squared distribution

        Returns
        ----------
        None, plt object displayed

        """

        #helper functions to shade the areas of given probabilities
        def _left_fill_helper(fill_to):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_to,
                              where=(self.x_range < fill_to), color='red', alpha=0.3)

        def _right_fill_helper(fill_to):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_to,
                              where=(self.x_range > fill_to), color='red', alpha=0.3)

        #plotting the function over the range of values
        plt.title(self.__repr__())
        plt.plot(self.x_range, self.pdf(),linestyle='dashed', color='red',linewidth=3)

        #reassigning some values from args to use in probability_calc()
        self.two_tail = two_tail
        self.left_cv = left_cv
        self.right_cv = right_cv

        #fill under the curve based on probability selection type
        if cv_probability==False:
            _right_fill_helper(self.df)
        elif (two_tail==True) & (cv_probability==True):
            _left_fill_helper(left_cv)
            _right_fill_helper(right_cv)

        plt.tight_layout()
        plt.show()

    def probability_calc(self):

        """
        This calculates the probability to the RIGHT of the critical value by
        integrating the area under the distribution from the critical value to
        infinity.

        Parameters
        ----------
        Self

        Returns
        ----------
        str, f string with self.probability if cv_probability = False
        
        str, f string with self.left_probability and self.right_probability
        if cv_probability = True

        """

        #function to integrate, pdf of Chi Squared
        f = lambda x: (1/(m.gamma(self.df/2)*2**(self.df/2)))*x**((self.df/2)-1)*m.e**(-x/2)

        #calculate the probabilities.
        if self.two_tail == False:
            self.right_probability, self.right_error_est = integrate.quad(f,self.df,np.inf)
            return ( f'P(X > df) is {round(self.right_probability,5)} with an error estimate of '
            f'{round(self.right_error_est,5)}')
        else:
            self.left_probability, self.left_error_est = integrate.quad(f, 0, self.left_cv) #left tail
            self.right_probability, self.right_error_est = integrate.quad(f, self.right_cv, np.inf) #right tail

            self.total_probability = self.left_probability + self.right_probability
            self.total_error = self.left_error_est + self.right_error_est

            return ( f'P(X < {self.left_cv}) is {round(self.left_probability,5)} '
                    f'and P(X > self.right_cv) is '
                    f'{round(self.right_probability,5)}. Total probability is '
                    f'{round(self.total_probability,5)} with total error estimate '
                    f'{round(self.total_error)}')

class t_rv:

    """
    Class for a random variable with a t-distribution and v degrees of freedom

    t_rv(deg_freedom, crit_value=0.0)
    
    Notes
    ----------
    As degrees of freedom increases to infinity, the t distribution approximates
    a standard normal distribution. You may notice that with >171 degrees of
    freedom, the math.gamma function returns a range error as this is a very
    large number and exceeds the Python-allowed data type limit.

    References
    ----------
    [1] Sahoo, Prasanna. "Probability and Mathematical Statistics",
    pp 396 (2008).
    
    """

    def __init__(self, df, crit_value=0.0):
        if df > 0:
            self.df = df
        else:
            raise ValueError('Degrees of freedom must be > 0')

        if df >= 3:
            self.mean = 0
            self.variance = (self.df / (self.df - 2))
        else:
            raise ValueError('E(X) DNE for df = 1 and Var(X) DNE for df = {1,2}')

        self.crit_value = float(crit_value)
        self.x_range = np.linspace(-2*self.df, 2*self.df, 2000)

    def __repr__(self):
        return f"t distribution with {self.df} degrees of freedom and critical value {self.crit_value}"

    def pdf(self):

        """
        this is the probability density function (pdf) of a t distribution with
        v degrees of freedom. 
        
        Parameters
        ----------
        Self

        Returns
        ----------
        numpy.ndarray, pdf of a t distribution evaluated over the range of x
        values for plotting purposes
        
        Notes
        ----------
        To check that it is, in fact, a pdf, the y values must integrate to 1.
        """

        return (m.gamma((self.df+1)/2) / (m.sqrt(m.pi * self.df) * 
                m.gamma(self.df / 2) * (1 + ((self.x_range**2)/self.df))**
                ((self.df + 1) / 2)))

    def plot_pdf(self, cv_probability=True, two_tail=False):

        """
        This function takes a given t random variable, uses the pdf that
        was previously calculated, and plots it.

        Parameters
        ----------
        cv_probability : bool, default True. the critical value probability 
        that determines the value the plot is shaded from. If False, shades 
        from the df to positive infinity. If True, shades from the critical 
        value.
        
        two_tail : bool, default False. If true, will fill lower and upper 
        tails 

        Returns
        ----------
        None, plt object displayed
        """

        def _right_fill_helper(fill_from):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_from, 
                              where=(self.x_range>fill_from), color='purple',
                              alpha=0.3)

        def _left_fill_helper(fill_to):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_to, 
                              where=(self.x_range<fill_to), color='purple',
                              alpha=0.3)

        plt.title(self.__repr__())
        plt.plot(self.x_range, self.pdf(),
                 linestyle='dashed', color='purple',linewidth=3)

        if (cv_probability==True) & (two_tail==False):
            _right_fill_helper(self.crit_value)
        elif (cv_probability==True) & (two_tail==True):
            _right_fill_helper(self.crit_value)
            _left_fill_helper(-self.crit_value)
        else:
            raise ValueError('please check the default arguments')

        plt.tight_layout()
        plt.show()


    def probability_calc(self, two_tail=False):

        """
        This calculates the probability to the RIGHT of the critical value by
        integrating the area under the distribution from the critical value
        to infinity.
        
        Parameters
        -----------
        two_tail : bool, default False. If two_tail is True, it will simply multiply the
        probability by 2. This is not too flexible yet, but will work for basic cases.

        Returns:
        -----------
        str, f string with probability
        """

        f = lambda x: (m.gamma((self.df+1)/2) / (m.sqrt(m.pi * self.df) * 
                       m.gamma(self.df / 2) * (1 + ((x**2)/self.df))**
                       ((self.df + 1) / 2)))
        self.probability, self.error_est = integrate.quad(f,self.crit_value,np.inf)
        
        if two_tail == False:
            return (f"P(X>crit_val) is {round(self.probability,5)} "
                   f"with an error estimate of {round(self.error_est,5)}")
        elif two_tail == True:
            return (
            f'P(X < {-self.crit_value}) & P(X > {self.crit_value}) is '
            f'{2*round(self.probability,5)} with an error estimate of '
            f'{round(self.error_est,5)}'
            )


class F_rv:

    """
    Class for a random variable with an F distribution with v_1 and v_2
    degrees of freedom with x > 0, v_2 >= 5. If v < 5, the Var(X) DNE and if
    v_2 < 3, E(X) DNE.

    F_rv(v_1, v_2, crit_value=0.0)

    Notes
    ----------
    As degrees of freedom increases to infinity, the F distribution approximates
    a standard normal distribution. You may notice that as degrees of
    freedom grows large, the math.gamma function returns a range error
    as this is a very large number and exceeds the Python-allowed data type
    limit.

    References
    ----------
    [1] Sahoo, Prasanna. "Probability and Mathematical Statistics",
    pp 401 (2008).
    
    
    """

    def __init__(self, v_1, v_2, left_cv=0.0, right_cv=1.0):

        if v_1 >0:
            self.v_1 = v_1
        else:
            raise ValueError('v_1 must be > 0')

        if v_2 >= 5:
            self.v_2 = v_2
            self.mean = self.v_2 / (self.v_2 - 2)
            self.variance = (2*self.v_2**2 * (self.v_1 + self.v_2 -2)) \
                            /((self.v_1)*(self.v_2 - 2)^2 * (self.v_2 -4))
        else:
            if v_2 < 3:
                raise ValueError('with v_2 < 3, E(X) DNE')
            else:
                raise ValueError('with v_2 < 5, Var(X) DNE')

        self.left_cv = float(left_cv)
        self.right_cv = float(right_cv)
        self.x_range = np.linspace(0, 2*self.v_2, 2000)

    def __repr__(self):
        return f"""F(v_1,v_2) distribution with v_1={self.v_1} , v_2={self.v_2}
        degrees of freedom, mean {round(self.mean,2)}, left critical value
        {self.left_cv} and right critical value {self.right_cv}"""

    def pdf(self):

        """
        this is the probability density function (pdf) of an F distribution with
        v_1 and v_2 degrees of freedom.

        Parameters
        ----------
        Self

        Returns
        ----------
        numpy.ndarray, pdf of an F distribution evaluated over the range of x
        values for plotting purposes

        Notes
        ----------
        To check that it is, in fact, a pdf, the y values must integrate to 1.
        """

        return (m.gamma((self.v_1 + self.v_2) / 2) * (self.v_1 / self.v_2)**(self.v_1 / 2) * self.x_range**((self.v_1 /2) -1)) \
        / (m.gamma(self.v_1 / 2) * m.gamma(self.v_2 / 2) * (1 + (self.v_1 /self.v_2)*self.x_range)**((self.v_1 + self.v_2) / 2))

    def plot_pdf(self, cv_probability=True, two_tail=False):

        """
        This function takes a given F random variable, uses the pdf that
        was previously calculated, and plots it.
        
        Parameters
        ----------
        cv_probability : bool, default False. the critical value probability 
        that determines the value the plot is shaded from. If False, shades 
        from the mean to positive infinity. If True, shades from the critical 
        value.
        
        two_tail : bool, default False. If true, will fill both lower and 
        upper tails 

        Returns
        ----------
        None, plt object displayed 
        """

        def _right_fill_helper(fill_from):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_from,
                              where=(self.x_range > fill_from),
                              color='forestgreen', alpha=0.3)
            
        def _left_fill_helper(fill_to):
            plt.fill_betweenx(self.pdf(), self.x_range, x2=fill_to,
                              where=(self.x_range < fill_to),
                              color='forestgreen', alpha=0.3)

        #plotting fn over the x range
        plt.title(self.__repr__())
        plt.plot(self.x_range, self.pdf(),linestyle='dashed', color='forestgreen',linewidth=3)

        if (cv_probability==True) & (two_tail==False):
            _right_fill_helper(self.right_cv)
        elif (cv_probability==True) & (two_tail==True):
            _right_fill_helper(self.right_cv)
            _left_fill_helper(self.left_cv)
        else:
            raise ValueError('please check the default arguments')

        plt.tight_layout()
        plt.show()

    def probability_calc(self,two_tail=False):

        """
        This calculates the probability to the RIGHT of the critical value by
        integrating the area under the distribution from the critical value
        to infinity.

        Parameters
        ----------
        self
        
        two_tail : bool, used in combination with cv_probability will 
        plot and shade both tails of an F distribution 

        Returns
        ----------
        str, f string with probability
        """

        f =  lambda x: ((self.v_2**(self.v_2/2) * self.v_1**(self.v_1/2)
                         * x**(self.v_1/2 -1))/
                        ((self.v_2 +self.v_1*x)**((self.v_1 + self.v_2)/2) *
                        beta(self.v_1/2, self.v_2/2)))
        
        #calculate the probabilities.
        if two_tail == False:
            self.right_probability, self.right_error_est = integrate.quad(f,self.right_cv,np.inf)
            return ( f'P(X > {self.right_cv}) is {round(self.right_probability,5)} with an error estimate of '
            f'{round(self.right_error_est,5)}')
        else:
            self.left_probability, self.left_error_est = integrate.quad(f, 0, self.left_cv) #left tail
            self.right_probability, self.right_error_est = integrate.quad(f, self.right_cv, np.inf) #right tail

            self.total_probability = self.left_probability + self.right_probability
            self.total_error = self.left_error_est + self.right_error_est

            return ( f'P(X < {self.left_cv}) is {round(self.left_probability,5)} '
                    f'and P(X > {self.right_cv}) is '
                    f'{round(self.right_probability,5)}. Total probability is '
                    f'{round(self.total_probability,5)} with total error estimate '
                    f'{round(self.total_error)}')
                