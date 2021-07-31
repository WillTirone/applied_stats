import math as m
import random as r
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

# based on example on p. 2 of hyp testing notes
class gen_test:

        def __init__(self, data, H0):

            """
            Initialize a general hypothesis test class

            Parameters
            ----------
            data : array-like, a list of data to draw from. I based this on
            something simple like the classic probability problem of drawing
            marbles from a bag and seeing how many were of a certain color.

            H0 : string, the null hypothesis to test. This does not have to be
            anything in particular as of right now.
            """

            self.data = list(data)
            self.H0 = H0

        def run_test(self, n, counter, accept_left, accept_right):

            """
            Run a general hypothesis test.

            Parameters
            ----------
            n : int, the number of samples to draw

            counter : str, the 'object' you want to count. For example, with a
            data set like ['R', 'B', 'R'] for red and blue marbles, to count
            red marbles, counter='R'

            accept_left : int, the left bound of the acceptance region

            accept_right : int, the right bound of the acceptance region

            Returns:
            ----------
            string, the decision made based on the sample drawn
            """

            ## BUG: for some reason, fails if sample size is 1 
            sample = r.sample(self.data, n)
            acceptance_region = {accept_left, accept_right}
            rejection_region = (set(i for i in range(1, len(self.data)))
                                .symmetric_difference(acceptance_region))
            sample_count = sample.count(counter)

            if sample_count in acceptance_region:
                decision = f'Do not reject H0. Count is {sample_count}'
            if sample_count in rejection_region:
                decision = f'Reject the H0. Count is {sample_count}'

            return decision
