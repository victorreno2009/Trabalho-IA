import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("spiral.csv", delimiter=',')

c1, c2 = [-1, 1]
C = 2

plt.scatter(data[data[:,2]==c1,0],data[data[:,2]==c1,1],color='red',edgecolor='k', alpha=.5)
plt.scatter(data[data[:,2]==c2,0],data[data[:,2]==c2,1],color='blue',edgecolor='k', alpha=.5)

plt.show()