import pycosat as pc
import numpy as np
import matplotlib.pyplot as plt
import time

#not finished

k = 3
dlimiter = 3
limiter = 8
nsamp = 10

def get_parenth(N):
    a = set()
    li = []
    while len(li)<k:
        num = np.random.randint(N)+1
        if num not in a:
            a.add(num)
            li.append(((np.random.randint(2)*2)-1)*num)
    return li

def get_expr(M,N):

    return [ get_parenth(N) for _ in range(M)]

M = 5
x = []
y = []


for N in [5,7,9,11,13,15,88]:
    print(N)
    x.append(N/M)
    start = time.time()
    ratios = [np.sum([1 for _ in range(nsamp) if len(list(pc.itersolve(get_expr(M,N)))) > 0])/nsamp for _ in range(1)]
    end = time.time()
    y.append(end-start)


    plt.plot(x,y)

plt.savefig('res2.png')
