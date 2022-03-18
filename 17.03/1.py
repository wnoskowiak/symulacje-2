from operator import truediv
import pycosat as pc
import numpy as np

cnf = [[1,2,-3],[-1,-2,3],[-1,-2,-3],[-1,2,-3],[-1,2,3],[1,-2,-3]]
a = pc.solve(cnf)
print(a)

N = 15
M = 7
k = 2

def get_parenth():
    a = set()
    li = []
    while len(li)<k:
        num = np.random.randint(N)+1
        if num not in a:
            a.add(num)
            li.append(((np.random.randint(2)*2)-1)*num)
    return li

def get_expr(M):

    return [ get_parenth() for _ in range(M)]

print('k=2')
for _ in range(10):
    a = get_expr(M)
    print("system:")
    print(a)
    sol = pc.itersolve(a)
    if len(list(sol)):
        print('solution:')
        print(list(sol))
    else:
        print('no solution')
    print('')

k = 3

print('k=3')
for _ in range(10):
    a = get_expr(M)
    print("system:")
    print(a)
    sol = pc.solve(a)
    if list(sol):
        print('solution:')
        print(sol)
    print('')
