import pyco as pc

cnf = [[1,2,-3],[-1,-2,3],[-1,-2,-3],[-1,2,-3],[-1,2,3],[1,-2,-3]]
a = pc.solve(cnf)
print(a)