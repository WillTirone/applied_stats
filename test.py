import unittest
import Statistics_OOP as stats

class Test_Distributions(unittest.TestCase):

    def test_norm_instance(self):
        a = stats.Norm_rv(0,1)
        self.assertIsInstance(a, stats.Norm_rv)

    def test_norm_probability(self):
        a = stats.Norm_rv(0,1)
        a.probability_calc()
        self.assertAlmostEqual(a.probability, 0.5)
