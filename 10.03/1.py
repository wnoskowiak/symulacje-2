import numpy as np
import matplotlib.pyplot as plt

def neighbours(i):
    ix, iy = i
    return [ ( ix, (iy+1) % L ), ( ix, (iy-1) % L ),
    ( (ix+1) % L, iy ), ( (ix-1) % L, iy ) ]

def update(s, h_loc, i):
    s[i] = 1
    for j in neighbours(i):
        h_loc[j] += 2.0
    return

L = 100

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

def runner(R):
        # lattice of spins
        s = np.ones( (L, L), dtype=int ) * (-1)
        # recording of avalanches
        aval = np.zeros( (L, L), dtype=int )
        # random magnetic fields
        h_rnd = np.random.randn(L, L) * R
        # ... and the local fields
        h_loc = np.ones( (L, L), dtype=int ) * (-4.0) + h_rnd
        return avalan(s, h_loc)

res = {r: np.average([runner(r) for _ in range(1000)]) for r in [0.7,0.9,1.4]}

print(res)


# while np.sum(s) != L ** 2:
#     aval = np.zeros( (L, L), dtype=int )
#     i_trig = np.unravel_index(np.argmax(h_loc + (s+1)*(-100)), h_loc.shape)
#     H = - h_loc[i_trig]
#     d = [i_trig]
#     aval = np.zeros( (L, L), dtype=int )
#     while d:
#         curr = d.pop(0)
#         if s[curr] == -1:
#             update(curr)
#             aval[curr] = 1
#             for elem in neighbours(curr):
#                 if (h_loc[elem] + H > 0) and (s[elem] == -1):
#                     d.append(elem) 
#     print(np.sum(aval))


#plt.imshow(aval,interpolation='none',cmap=cm.gist_rainbow)

