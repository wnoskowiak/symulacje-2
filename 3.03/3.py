import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(loc= 15,scale = 100,size = 10000)

#1

print(np.std(a))
print(np.average(a))

plt.hist(a, bins = 25)

#2

N = 10000

P=10

a = np.array([np.random.uniform(low= -1,size = N) for _ in range(P)])

b = a.sum(axis = 0)

xnew = 15 + 100*b*np.sqrt(3./P)
print(xnew)

plt.hist(xnew, bins = 25)

plt.show()
