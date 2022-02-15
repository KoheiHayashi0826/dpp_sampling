import os 

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import torch

from dppy.beta_ensembles import HermiteEnsemble

def kernel(n, x, y):
    out_list = []
    Hn0 = special.hermite(n, monic=True)
    Hn1 = special.hermite(n-1, monic=True)
    const = np.prod( (2*np.ones(n-1) / np.arange(1,n)) )
    if x!=y:
        out = const * (Hn0(0.5*x) * Hn1(0.5*y) - Hn1(0.5*x) * Hn0(0.5*y)) / (0.5*x - 0.5*y) 
    else:
        out = const * (n*Hn1(0.5*x)**2 - x*Hn1(0.5*x) + 2*Hn0(0.5*x)**2 )
    return out

print(kernel(10, 1, 1))
        
        




"""
# sampling by dppy
hermite = HermiteEnsemble(beta=2)
a = hermite.sample_full_model(size_N=500) # return as ndarray
x = np.linspace(0, 10, 500)
#plt.plot(x, a)
#plt.show()
"""
