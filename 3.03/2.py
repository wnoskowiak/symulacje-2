import numpy as np

n = 4000
A = np.zeros((n,n))
x = np.zeros(n)

for i in range(n):
    x[i] = i/2.0 # some vector
    for j in range(n): # ... and a matrix
        A[i,j] = i + j + (i+1) % (j+1)
print( x )
print( A )

b = np.dot(A, x)

y = np.linalg.solve(A,b)

print(x)

print(np.linalg.norm(x-y))