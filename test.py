import unittest
import numpy as np
import Statistics_OOP as stats

class Test_Distributions(unittest.TestCase):

    def test_norm(self):

        #test an instance
        a = stats.Norm_rv(0,1)
        self.assertIsInstance(a, stats.Norm_rv)

        #test the probability calculation
        a.probability_calc()
        self.assertAlmostEqual(a.probability, 0.5)

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

        #test some attributes
        self.assertEqual(c.df, 5)
        self.assertEqual(c.mean, 0)
        self.assertEqual(c.variance, 5/3)

    def test_F(self):
        pass


if __name__ == '__main__':
    unittest.main()
