import numpy as np
from scipy.linalg import kron
import numpy as np
import matplotlib.pyplot as plt

sz = np.array([[1.,0.],[0.,-1.]])
sx = np.array([[0.,1],[1, 0.]])
one = np.eye(2)

szi = [kron( sz, kron(one, one)), kron( one, kron(sz, one)), kron( one, kron(one, sz))]
sxi = [kron( sx, kron(one, one)), kron( one, kron(sx, one)), kron( one, kron(one, sx))]

j12 = -1.1
j13 = -2.1
j23 = -3.8

h= [0.6,0,0]

j = [[0,j12,j13],
     [0,0,j23],
     [0,0,0]]

h0 = -np.sum(sxi, axis=0)
h1 = -np.sum([np.sum([(j[i][k])*(szi[i]@szi[k]) for k in range(3)], axis=0) for i in range(3)],axis=0) - h[0]*szi[0]


def get_ham(lamb):
    return (1-lamb)*h0 + lamb*h1

def get_delta(lamb):
    ham = get_ham(lamb)
    eig,eigv = np.linalg.eig(ham)
    E_0 = np.min(eig)
    eigv0 = [d[0][0] for d in (eigv[:,np.where(eig == E_0)])]
    s1v = eigv0@szi[0]@eigv0
    s2v = eigv0@szi[1]@eigv0
    s3v = eigv0@szi[2]@eigv0
    eig = eig[eig != E_0]
    E_1 = np.min(eig)
    return E_1-E_0,s1v,s2v,s3v

dx = 0.0001
x = np.arange(0,1,dx)
wyn = np.transpose(np.array([get_delta(l) for l in x]))
print(wyn)
y = wyn[0]
print(y)
plt.plot(x,y)
plt.show()

integral = dx*np.sum(1/y**2)
print(integral)
for i in range(1,4):
    y = wyn[i]
    plt.plot(x,y)
plt.show()

