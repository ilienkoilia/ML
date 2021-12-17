import numpy as np
import random
import matplotlib.pyplot as plt

def Checking_circle(list_points,a,d,n_points): 
    sum_cercle = 0
    for i in range(n_points):
        sum = 0
        for j in range(d):
            sum = sum + (list_points[i][j])**2 
        if sum <= (a/2)**2: 
            sum_cercle +=1
    return sum_cercle/n_points
        


list_points = []
xmin = 1 
xmax = 10 

a = 100 
n_points = 9100 

xlist = np.linspace(xmin, xmax, xmax-xmin+1) 
ylist = []


for d in range(xmin,xmax+1):
    list_points = []
    for i in range(n_points):
        temp_points = []
        for j in range(d):
            temp_points.append(random.randint(-a/2, a/2)) 
        list_points.append(temp_points)
    ylist.append(Checking_circle(list_points,a,d,n_points)) 

plt.plot(xlist, ylist)
plt.show()