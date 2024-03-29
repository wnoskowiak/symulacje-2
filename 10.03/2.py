from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def neighbours(i):
    ix, iy = i
    return [ ( ix, (iy+1) % L ), ( ix, (iy-1) % L ),
    ( (ix+1) % L, iy ), ( (ix-1) % L, iy ) ]

def update(s, h_loc, i):
    s[i] = 1
    for j in neighbours(i):
        h_loc[j] += 2.0
    return

def avalan(s, h_loc):
    aval = np.zeros( (L, L), dtype=int )
    i_trig = np.unravel_index(np.argmax(h_loc + (s+1)*(-100)), h_loc.shape)
    H = - h_loc[i_trig]
    d = [i_trig]
    aval = np.zeros( (L, L), dtype=int )
    while d:
        curr = d.pop(0)
        if s[curr] == -1:
            update(s, h_loc,curr)
            aval[curr] = 1
            for elem in neighbours(curr):
                if (h_loc[elem] + H > 0) and (s[elem] == -1):
                    d.append(elem) 
    return np.sum(aval)

L = 300
for R in [0.9, 1.4, 2.1]:
    # lattice of spins
    s = np.ones( (L, L), dtype=int ) * (-1)
    # recording of avalanches
    aval = np.zeros( (L, L), dtype=int )
    # random magnetic fields
    h_rnd = np.random.randn(L, L) * R
    # ... and the local fields
    h_loc = np.ones( (L, L), dtype=int ) * (-4.0) + h_rnd
    count = 0

    Hs = []
    Ms = []

    while np.sum(s) != L ** 2:
        count += 1 
        i_trig = np.unravel_index(np.argmax(h_loc + (s+1)*(-100)), h_loc.shape)
        H = - h_loc[i_trig]
        d = [i_trig]
        while d:
            curr = d.pop(0)
            if s[curr] == -1:
                aval[curr] = count
                update(s, h_loc,curr)
                for elem in neighbours(curr):
                    if (h_loc[elem] + H > 0) and (s[elem] == -1):
                        d.append(elem) 

        Hs.append(H)
        Ms.append(np.sum(s)/L**2)

    print(aval)
    plt.plot(Ms,Hs,label = f'R={R}')
    plt.title(f'H(M)')
    plt.xlabel('M')
    plt.ylabel('H')
    plt.xlim(-1,1)
    plt.ylim(-3,3)
plt.legend(frameon=False, loc='lower center', ncol=2)
plt.show()
    



