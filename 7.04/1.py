from calendar import monthrange
from typing import NoReturn
from xmlrpc.client import Boolean
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 100
l_0 = 1
eps = 0.25
#s = 0.5
k = 1000
k_b = 100
kt = 0.001

L = N*l_0

def get_energy(l,c,r):
    len_l = np.sqrt(np.sum((l-c)**2))
    len_r = np.sqrt(np.sum((r-c)**2))
    E_compressive = .5*k*((l_0-len_l)**2+(l_0-len_r)**2)
    angle_cos = np.dot((c-l),(c-r))/(len_l*len_r)
    E_angular = k_b*(1+angle_cos)
    return E_compressive+E_angular

def get_total_energy(arra):
    En = 0
    for i in range(1,len(chain[0])-1):
        En+= get_energy(
            np.array([chain[0][i-1],chain[1][i-1]]),
            np.array([chain[0][i],chain[1][i]]),
            np.array([chain[0][i+1],chain[1][i+1]])
        )
    return En


import itertools
chain = np.append(np.arange(0,L,l_0),np.zeros(N)).reshape((2,N))

energy = []

chain[0][0] += .5*eps*L
chain[0][-1] -= .5*eps*L 

print(chain)

for s in [0.01,0.005]:
    print(s)
    for rt in range(10000):
        if((rt%300)==0):
            energy.append(get_total_energy(chain))
            plt.plot(chain[0],chain[1])
            plt.title(f"N = {N}, k={k}, l_0 = {l_0}, kb= {k_b}, kt= {kt}")
            plt.savefig(f"fig{rt}_{s}.png")
            plt.clf()
        for i in range(1,len(chain[0])-1):

            #calculating current 
            delta_x = np.random.uniform(-s,s)
            delta_y = np.random.uniform(-s,s)
            E_old = get_energy(
                np.array([chain[0][i-1],chain[1][i-1]]),
                np.array([chain[0][i],chain[1][i]]),
                np.array([chain[0][i+1],chain[1][i+1]])
            )
            E_new = get_energy(
                np.array([chain[0][i-1],chain[1][i-1]]),
                np.array([chain[0][i]+delta_x,chain[1][i]+delta_y]),
                np.array([chain[0][i+1],chain[1][i+1]])
            )
            if i>1:
                E_old += get_energy( 
                    np.array([chain[0][i-2],chain[1][i-2]]),
                    np.array([chain[0][i-1],chain[1][i-1]]),
                    np.array([chain[0][i],chain[1][i]])
                )
                E_new += get_energy(
                    np.array([chain[0][i-2],chain[1][i-2]]),
                    np.array([chain[0][i-1],chain[1][i-1]]),
                    np.array([chain[0][i]+delta_x,chain[1][i]+delta_y])
                )
            else:
                E_old += get_energy( 
                    np.array([chain[0][i-1]-l_0,0]),
                    np.array([chain[0][i-1],chain[1][i-1]]),
                    np.array([chain[0][i],chain[1][i]])
                )
                E_new += get_energy(
                    np.array([chain[0][i-1]-l_0,0]),
                    np.array([chain[0][i-1],chain[1][i-1]]),
                    np.array([chain[0][i]+delta_x,chain[1][i]+delta_y])
                )

            if i<N-2:
                E_old += get_energy(
                    np.array([chain[0][i],chain[1][i]]),
                    np.array([chain[0][i+1],chain[1][i+1]]),
                    np.array([chain[0][i+2],chain[1][i+2]])
                )
                E_new += get_energy(
                    np.array([chain[0][i]+delta_x,chain[1][i]+delta_y]),
                    np.array([chain[0][i+1],chain[1][i+1]]),
                    np.array([chain[0][i+2],chain[1][i+2]])
                )
            else:
                E_old += get_energy(
                    np.array([chain[0][i],chain[1][i]]),
                    np.array([chain[0][i+1],chain[1][i+1]]),
                    np.array([chain[0][i+1]+l_0,0])
                )
                E_new += get_energy(
                    np.array([chain[0][i]+delta_x,chain[1][i]+delta_y]),
                    np.array([chain[0][i+1],chain[1][i+1]]),
                    np.array([chain[0][i+1]+l_0,0])
                )


            Delta_E = E_new-E_old
            #print(Delta_E)

            random_number = np.random.uniform(0,1)

            if (random_number<min(1,(np.exp(-Delta_E/kt)))):
                chain[0][i] += delta_x
                chain[1][i] += delta_y


plt.plot(chain[0],chain[1])
plt.title(f"N = {N}, k={k}, l_0 = {l_0}, kb= {k_b}, kt= {kt}")
plt.show()
plt.clf()
plt.plot(energy)
plt.show()
            


