from unittest import result
import numpy as np

mutationRate= 0.01

Nbits = 15

def f(x,y):
    return (((x*x)+y-11)**2)+(x+(y*y)-7)**2

def getXandY(item):
    mx = (2**(Nbits))
    x = (8*int("".join(str(a) for a in item[:Nbits]), 2)/mx)-4
    y = (8*int("".join(str(a) for a in item[Nbits:]), 2)/mx)-4
    return[x,y]

def valueFunc(item):
    a = np.power(f(*getXandY(item))+1e-6,-1)
    return a
    

def getProbArray(population):
    result = [0]
    result.extend(result[-1] + valueFunc(elem) for elem in population)
    result /= result[-1]
    return result

def getRandomParent(probArray):
    random = np.random.uniform()
    for i in range(1,len(probArray)):
        if (probArray[i]>random):
            return i-1

def makeBaby(parent1, parent2):
    random = np.random.randint(0,high = len(parent1)-1)
    part1 = [parent1[i] for i in range(random)]
    part2 = [parent2[i] for i in range(random,len(parent1))]
    result = part1+part2
    if(mutationRate>np.random.uniform()):
        muatation  = np.random.randint(0,high = len(parent1))
        result[muatation] = (result[muatation]+1)%2 
    return result
    
def getPatents(population, probArray):
    parent1 = getRandomParent(probArray)
    parent2 = getRandomParent(probArray)
    while(parent1==parent2):
        parent2 = getRandomParent(probArray)
    return [population[parent1], population[parent2]]

population = np.array([np.random.choice([0, 1], size=(2*Nbits,)) for _ in range(200)])

for _ in range(250):
    probArray = getProbArray(population)
    population = np.array([makeBaby(*getPatents(population,probArray)) for _ in range(len(population))])

print((getXandY(population[np.argmax([valueFunc(elem) for elem in population])])))
print(f(*getXandY(population[np.argmax([valueFunc(elem) for elem in population])])))

