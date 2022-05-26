import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sympy import lucas
import itertools

alpha = 1
beta = 1
rho = 0.004
Q = 0.07

npoints = 45
np.random.seed(1234) #to reproduce meshes remove for production
points = np.random.randint(0,17, size=(npoints, 2))
points = np.unique(points, axis=0)

"""
Dirty way to ensure that there are npoints unique
points in the mesh
"""
max_iter = 100
itera = 0
while points.shape[0]-npoints<0:
    next_points = np.random.randint(0,10, size=(npoints-points.shape[0], 2))
    points = np.append(points, next_points, axis=0)
    points = np.unique(points, axis=0)
    print(next_points.shape[0], points.shape[0])
    itera += 1
    if itera>max_iter:
        break
npoints = points.shape[0]

D = Delaunay(points)
numPoints = [np.array(a) for a in D.points]
simplices = D.simplices
adje = np.zeros((npoints,npoints))
weights = np.zeros((npoints,npoints))
phero = np.ones((npoints,npoints))

def distance(list):
    # print(len(list))
    return sum(np.linalg.norm(numPoints[list[i - 1]] - numPoints[list[i]]) for i in range(1, len(list)))

def simplifyPath(list):
    visited = -np.ones(npoints, dtype=int)
    list = np.array(list)
    for i in range(len(list)):
        if visited[list[i]] != -1:
            for j in range(visited[list[i]],i):
                list[j] = -1
        visited[list[i]] = i
    return list[np.where(list>= 0)]

def getVertex(adj, weights,phero, current):
    probs = [0]
    verts = []
    for i in range(npoints):
        if (adj[current][i]):
            temp = (weights[current][i]**alpha)*(phero[current][i]**beta)
            probs.append(probs[-1] + temp)
            verts.append(i)
    probs /= probs[-1]
    rand = np.random.uniform(0, 1)
    for i in range(1,len(probs)):
        if rand<probs[i]:
            return verts[i-1]

def getRandomPath(adj, start = 0, stop = npoints-1):
    point = start
    path = [start]
    while point != stop:
        path.append(getVertex(adj, weights,phero, point))
        point = path[-1]
    return simplifyPath(path)

print(D.points)

for i, list in itertools.product(range(npoints), D.simplices):
    if i in list:
        for elem in list:
            if elem != i:
                adje[i][elem] = 1

print(adje)
phero = np.copy(adje)
for i in range(npoints):
    for j in range(npoints):
        if adje[i][j] == 1:
            weights[i][j] = 1/distance([i, j])

print(weights)
plt.pcolormesh(weights, cmap = 'inferno')
plt.show()
def step():
    paths = [getRandomPath(adje) for _ in range(150)]
    deltas = np.zeros((npoints,npoints))
    for path in paths:
        len = distance(path)
        incre = Q/len
        for i in range(1, path.size):
            deltas[path[i-1]][path[i]] += incre
    return deltas

for i in range(100):
    phero = phero*(1-rho) + step()
pts=[]
current_point_idx = 0
while (current_point_idx != npoints-1):
    pts.append(D.points[current_point_idx])
    current_point_idx = np.argmax(phero[current_point_idx])
pts.append(D.points[npoints-1])
pts = np.transpose(pts)
print(pts)

plt.pcolormesh(phero, cmap = 'inferno')
plt.show()

plt.figure()
plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')
plt.plot(pts[0],pts[1])
plt.show()

