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

for R in [0.9,1.2,2.1]:
    for _ in range(5):
        L = 300
        # lattice of spins
        s = np.ones( (L, L), dtype=int ) * (-1)
        # recording of avalanches
        aval = np.zeros( (L, L), dtype=int )
        # random magnetic fields
        h_rnd = np.random.randn(L, L) * R
        # ... and the local fields
        h_loc = np.ones( (L, L), dtype=int ) * (-4.0) + h_rnd
        count = 1

        sizes= []

        while np.sum(s) != L ** 2:
            aval = np.zeros( (L, L), dtype=int )
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
            sizes.append(np.sum(aval))


    counts, bins = np.histogram(sizes)
    print(counts, bins)
    plt.plot(bins[:-1], counts, label =f'R={R}')
    plt.xscale('log')
    plt.yscale('log')
plt.legend(frameon=False, loc='upper right', ncol=2)
plt.show()

    



