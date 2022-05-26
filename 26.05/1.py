import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

eta = 0.001

training = read_csv("26.05\\train_data.out").to_numpy()
# validation = read_csv("26.05\\validation_data.out").to_numpy()
training = np.insert(training,0,1,axis=1)
print(training)
x = training[:,[0,1,2]]
y = (training[:,3]+1)/2
w = np.array([1,0.4,0.4])

for _ in range(300):
    for i in range(len(x)):
        y_hat = (1 if np.dot(x[i],w)>0 else 0)
        w[0] += eta*((y[i])-y_hat)*x[i][0]
        w[1] += eta*((y[i])-y_hat)*x[i][1]
        w[2] += eta*((y[i])-y_hat)*x[i][2]

print(w)

valid = read_csv("26.05\\validation_data.out").to_numpy()
# validation = read_csv("26.05\\validation_data.out").to_numpy()
valid = np.insert(valid,0,1,axis=1)
x = valid[:,[0,1,2]]
y = (valid[:,3]+1)/2

print(np.sum([(y[i] - (1 if np.dot(x[i],w)>0 else 0))**2 for i in range(len(valid))]))

def get_y_val(x_val):
    return -(x_val *w[1] + w[0])/w[2]

plt.scatter(x[:,1],x[:,2],c=y)
x_val = np.arange(-5,3,0.001)
y_val = get_y_val(x_val)
plt.plot(x_val,y_val)
plt.show()