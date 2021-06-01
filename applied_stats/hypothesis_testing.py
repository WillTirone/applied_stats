import math as m
from random import random
import numpy as np

import continuous_distributions as st

class norm_hyp(st.Norm_rv):

    def __init__(self, mean, variance, H0, HA, type):

        # might not want a mean if that's what we're testing,
        # we wouldn't know the mean in advance
        super().__init__(mean, variance)

        #new values of subclass
        self.std_dev = m.sqrt(variance)
        self.H0 = H0
        self.HA = HA
        self.type = type # simple or compound test

    def z_score(self, x):

        """calculate a z score"""

        self.z = (x - self.mean) / (self.variance)
        return self.z

class gen_test:

        def __init__(self, data, H0):
            self.data = np.array(data)
            self.H0 = H0
            self.results = {}

        def run_test(self):
            self.sample = random.sample(self.data, n)

            #redoing marble example as a base case
            self.acceptance_region = [0,4]
            self.rejection_region = [1,2,3]


        #TODO: every time we test something, add the result to a dictionary
        #TODO: look at decorators, or things to juice up classes
