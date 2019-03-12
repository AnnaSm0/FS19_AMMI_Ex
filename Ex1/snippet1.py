import numpy as np

def f(x, y, z, noise=1e-4):
    return np.array([24.5+15*x + noise*np.random.normal(),
                     1.2+30*y+10*z + noise*np.random.normal()])

# generate data
n = 100
X = np.random.rand(n,3)
Y = f(X[:,0], X[:,1], X[:,2]).transpose()
