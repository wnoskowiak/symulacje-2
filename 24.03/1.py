import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N=1000

def quad(x,a,b,c):
    return np.multiply(np.power(x,2),a) +np.multiply(x,b) + c

def lin(x,a,b):
    return np.multiply(x,a)+b

theta = np.zeros(N)
r = [np.array([0,0]) for _ in range(N)]

def get_random_vec():
    a = np.random.normal(0)
    b = np.random.normal(0)
    return np.array([a,b])

def get_theta(dt, mu, i):
    global theta
    old_theta = theta[i]
    theta[i] = theta[i] + np.sqrt(dt*mu)*np.random.uniform(-1,1) 
    return old_theta

def get_n(theta):
    return np.array([np.cos(theta),np.sin(theta)])

def get_r(dt, gamma, v, mu, alpha, i):
    global r
    r[i] = r[i] + (dt/gamma)*v*get_n(get_theta(dt, mu, i))+np.sqrt(dt*alpha)*get_random_vec()
    return r[i]

def get_r2(r):
    return r.dot(r)

mu = 10**(-1)
alpha = 10**(-3)
gamma = 10**(0)

time = 10**(2)
dt = 10**(-1)

times = [n*dt for n in range(int(time/dt))]

for v in [0,2,4,6,8,10]:
    traj = [[ get_r2(get_r(dt, gamma, v, mu, alpha, i)) for _ in range(int(time/dt))] for i in range(N)]

    traj = np.transpose(traj)

    res = [np.average(elem) for elem in traj]

    plt.plot(times,res, label=f'v={v}')

    y1 = res[:200]
    x1 = times[:200]

    y2 = res[-200:]
    x2 = times[-200:]

    coof,dump = curve_fit(quad,x1,y1)

    plt.plot(x1,quad(x1,*coof), '--', linewidth=3)

    coof,dump = curve_fit(lin,x2,y2)

    plt.plot(x2,lin(x2,*coof), '--', linewidth=3)



    theta = np.zeros(N)
    r = [np.array([0,0]) for _ in range(N)]



plt.legend()

plt.xlabel('time [100 ms]')
plt.ylabel('MSD')
plt.show()