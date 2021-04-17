import numpy as np
import mle 

# defining a random array and sample size for reproducible testing 
rng = np.random.default_rng(1905)
X_continuous = rng.random(100) 
X_discrete = np.round(X_continuous, 0)
n = len(X_continuous)

q = np.random.binomial(10, 1, 50)