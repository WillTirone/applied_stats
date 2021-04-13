import unittest
import math as m 

import numpy as np
from scipy import integrate
from scipy.special import beta

import stats_tools.continuous_distributions as stats
import stats_tools.mle as mle 

# defining a random array and sample size for reproducible testing 
rng = np.random.default_rng(1905)
X_continuous = rng.random(100) * 100 
n = len(X_continuous)

# testing distribution calculations / attributes / methods 
class Test_Distributions(unittest.TestCase):

    def test_norm(self):

        #test an instance
        a = stats.Norm_rv(0,1)
        self.assertIsInstance(a, stats.Norm_rv)

        #test the probability calculation
        a.probability_calc()
        self.assertAlmostEqual(a.probability, 0.5)
        
        #test that it is a pdf by integrating, it must = 1
        f = lambda x: ((1/(a.sigma*m.sqrt(2*m.pi)))*
                       m.e**((-1/2)*((x-a.mean)/a.sigma)**2))
        a.probability, a.error_est = integrate.quad(f,-np.inf, np.inf)
        self.assertAlmostEqual(a.probability, 1)
    
        #testing attributes
        self.assertEqual(a.mean,0)
        self.assertEqual(a.variance,1)
        self.assertTrue(a.variance < np.infty)

    def test_chisq(self):

        #test an instance
        b = stats.ChiSq_rv(4,crit_value=7)
        self.assertIsInstance(b, stats.ChiSq_rv)

        #test the probability calculation
        b.probability_calc()
        self.assertAlmostEqual(round(b.probability,5), .13589)
        
        #test that it is a pdf by integrating, it must = 1
        f = lambda x: ((1/(m.gamma(b.df/2)*2**(b.df/2)))
                       *x**((b.df/2)-1)*m.e**(-x/2))
        b.probability, b.error_est = integrate.quad(f,0,np.inf)
        self.assertAlmostEqual(b.probability, 1)

        #test some attributes
        self.assertEqual(b.df, 4)
        self.assertEqual(b.crit_value, 7)
        self.assertEqual(b.mean, 4)
        self.assertEqual(b.variance, 8)

    def test_t(self):

        #test an instance
        c = stats.t_rv(5,crit_value=1)
        self.assertIsInstance(c, stats.t_rv)

        #test the probability calculation
        c.probability_calc()
        self.assertAlmostEqual(round(c.probability,5), 0.18161)

        #test that it is a pdf by integrating, it must = 1
        f = lambda x: (m.gamma((c.df+1)/2) / (m.sqrt(m.pi * c.df) * 
                       m.gamma(c.df / 2) * (1 + ((x**2)/c.df))
                       **((c.df + 1) / 2)))
        c.probability, c.error_est = integrate.quad(f,-np.inf,np.inf)
        self.assertAlmostEqual(c.probability, 1)

        #test some attributes
        self.assertEqual(c.df, 5)
        self.assertEqual(c.mean, 0)
        self.assertEqual(c.variance, 5/3)

    def test_F(self):
        
        #test an instance 
        d = stats.F_rv(5, 5, 1.5)
        self.assertIsInstance(d, stats.F_rv)
        

        #test the probability calculation
        d.probability_calc()
        #self.assertAlmostEqual(round(d.probability,2), 0.33)

        #test that it is a pdf by integrating, it must = 1
        f =  lambda x: ((d.v_2**(d.v_2/2) * d.v_1**(d.v_1/2) * 
                         x**(d.v_1/2 -1))/
                        ((d.v_2 +d.v_1*x)**((d.v_1 + d.v_2)/2) * 
                        beta(d.v_1/2, d.v_2/2))) 
        d.probability, d.error_est = integrate.quad(f,0,np.inf)
        self.assertAlmostEqual(d.probability, 1)
        
        #test some attributes 
        self.assertEqual(d.v_1, 5)
        self.assertEqual(d.v_2, 5)
        self.assertEqual(round(d.mean,3), 1.667)
        self.assertEqual(round(d.variance,3), 30.769)

# testing the MLE module to ensure accurate calculations 
class Test_MLE(unittest.TestCase):

    def test_uniform(self):

        a = round(mle.uniform(X_continuous),4)
        self.assertEqual(a, 99.0877)

    def test_exponential(self): 
        
        b = round(mle.exponential(n, X_continuous),4)
        self.assertEqual(b, 52.1989)

    def test_normal(self): 
        
        c, d = mle.normal(n, X_continuous)
        self.assertEqual(round(c,4), 52.1989)
        self.assertEqual(round(d,4), 747.7962)

if __name__ == '__main__':
    unittest.main()
