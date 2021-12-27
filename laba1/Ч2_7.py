import numpy as np
import random
import matplotlib.pyplot as plt

def Checking_circle(list_points, a, d, n_points): 
    sum_cercle = np.sum(np.sum(list_points ** 2, axis=1) <= (a / 2) ** 2)
    return sum_cercle / n_points
        

list_points = []
xmin = 1 
xmax = 10 

a = 100 
n_points = 9100 

xlist = np.linspace(xmin, xmax, xmax - xmin + 1) 
ylist = []


for d in range(xmin, xmax+1):
    list_points = np.random.rand(n_points, d + 1) * a / 2
    ylist.append(Checking_circle(list_points, a, d, n_points)) 

plt.plot(xlist, ylist)
plt.show()