from turtle import position
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

N = 40
beta = 4
delta = 0.29
BDelta = beta/N

def pif(x):
    sigma = 1/(2*np.tanh(beta/2))
    return (1/(np.sqrt(2*3.14159265359*sigma)))*np.exp(-(x**2)/(2*sigma))

@jit(nopython=True)
def V(x):
    return -.5*(x)**2

@jit(nopython=True)
def rho_free(x,y,BD):
    return np.exp(-((x-y)**2)/(2*BD))

@jit(nopython=True)
def native_harmonic_path(positions):
    r = 0
    k = np.random.randint(0,N)
    km = (k-1)%(N-1)
    kp = (k+1)%(N-1)
    nk = positions[k]+ np.random.uniform(-delta,delta) 
    pia = rho_free(positions[km],positions[k],BDelta)*rho_free(positions[k],positions[kp],BDelta)*np.exp(BDelta*V(positions[k]))
    pib = rho_free(positions[km],nk,BDelta)*rho_free(nk,positions[kp],BDelta)*np.exp(BDelta*V(nk))
    gamma = pib/pia
    if(np.random.uniform(0,1)<gamma):
        r += 1
        positions[k] = nk
    return r,positions

@jit(nopython=True)
def MCS(positions):
    r = 0
    for _ in range(N):
        rt,positions = native_harmonic_path(positions)
        r+=rt
    return r,positions

positions = np.random.uniform(-delta,delta,N)
avg = np.zeros(100000)
var = np.zeros(100000)
xz = np.zeros(100000)
xnh = np.zeros(100000)
g = 0
for i in range(len(avg)):
    t,positions = MCS(positions)
    avg[i] = np.average(positions)
    var[i] = np.var(positions)
    xz[i] = positions[0]
    xnh[i] = positions[N//2]
    g += t 

print(positions)
print(g/4000000)
plt.plot(avg)
plt.axhline(y=np.average(avg), color='r', linestyle='-')
plt.show()
print(np.average(var))
plt.axhline(y=np.average(var), color='r', linestyle='-')
plt.plot(var)
plt.show()

x = np.arange(-2,2,0.001)
y = pif(x)
counts, bins = np.histogram(xz,density=True,bins=30)
plt.hist(bins[:-1], bins, weights=counts)
plt.plot(x,y)
plt.show()

counts, bins = np.histogram(xnh,density=True,bins=30)
plt.hist(bins[:-1], bins, weights=counts)
plt.plot(x,y)
plt.show()


