import numpy as np

# set random seed
np.random.seed()

def f(x,y):
    return 3 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2) \
           - 10 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2) \
           - 1 / 3 * np.exp( -(x + 1)**2 - y**2)


# random search optimization
radius = 1
num_iterations = 10000

# initial position between -3 and 3
x = (np.random.rand()-0.5)*6
y = (np.random.rand()-0.5)*6

value = f(x,y)
