from cmath import sqrt
import os
from traceback import print_tb 

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
import torch

from dppy.beta_ensembles import HermiteEnsemble


def kernel(n, x, y):
    Hn0 = special.hermite(n, monic=True)
    Hn1 = special.hermite(n-1, monic=True)
    const = np.prod( (2*np.ones(n-1) / np.arange(1,n)) )
    if x!=y:
        out = const * (Hn0(0.5*x) * Hn1(0.5*y) - Hn1(0.5*x) * Hn0(0.5*y)) / (0.5*x - 0.5*y) 
    else:
        out = const * (n*Hn1(0.5*x)**2 - x*Hn1(0.5*x) + 2*Hn0(0.5*x)**2 )
    return out
#print(kernel(10, 1, 1))


def kernel1(n, x, y):
    out = 1
    for i in range(1, n): # range(1,n) = [1,...,n-1]
        H = special.hermite(i, monic=True) # H_n_monic = 2^(-n) H_n
        const = np.prod( (2*np.ones(i) / np.arange(1,i+1)) )
        out += const * H(0.5*x) * H(0.5*y)
    return out


def kernel2(n, x, y):
    out = 1
    for i in range(1, n): # range(1,n) = [1,...,n-1]
        H = special.hermite(i, monic=True) # H_n_monic = 2^(-n) H_n
        const, scale = 2, 0.5
        out += H(scale*x) * H(scale*y) 
    return out


def pdf_dpp(K: torch.Tensor):
    n = K.shape[0]
    out = K[0, 0] / n
    for i in range(n-1):
        out *= torch.det(K[:i+1,:i+1]) / (torch.det(K[:i, :i]) * (n-i-1) )
    return out


n = 12
K = torch.zeros(n, n)
K1 = torch.zeros(n, n)
K2 = torch.zeros(n, n)
#x = torch.rand(n)
x = torch.randn(n) * 5
#print(x)
for i in range(n):
    for j in  range(n):
        K[i, j] = kernel(n, x[i], x[j])
        K1[i, j] = kernel1(n, x[i], x[j])
        K2[i, j] = kernel1(n, x[i], x[j])
#print(K, K1)
#print(K1-torch.t(K1))
print(torch.det(K), torch.det(K1), torch.det(K2))
print(pdf_dpp(K), pdf_dpp(K1), pdf_dpp(K2))





"""
# sampling by dppy
hermite = HermiteEnsemble(beta=2)
a = hermite.sample_full_model(size_N=500) # return as ndarray
x = np.linspace(0, 10, 500)
#plt.plot(x, a)
#plt.show()
"""
