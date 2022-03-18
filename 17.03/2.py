import pycosat as pc
import numpy as np
import matplotlib.pyplot as plt

k = 3
dlimiter = 3
limiter = 8
nsamp = 125

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



for N in [5,10,20]:
    ms = [m/N for m in range(dlimiter*N,limiter*N)]
    ratios = [np.sum([1 for _ in range(nsamp) if len(list(pc.itersolve(get_expr(m,N)))) > 0])/nsamp for m in range(dlimiter*N,limiter*N)]
    plt.plot(ms,ratios)

plt.savefig('res.png')
