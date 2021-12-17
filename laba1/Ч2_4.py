import numpy as np
import matplotlib.pyplot as plt
import pylab

def transformation_plot(p, array):
    for i in range(len(p)):
        p[i][0] =  p[i][0] * array[0][0] + p[i][1] * array[0][1]
        p[i][1] =  p[i][0] * array[1][0] + p[i][1] * array[1][1]
    return p



array = [
    [2,4],
    [1,5]]
p = [
    [2,4],
    [1,4],
    [4,-1]]

pylab.subplot (2, 1, 1)
for i in range(len(p)):
    plt.plot(p[i][0], p[i][1],'ro')
    plt.annotate(str(i+1), xy=(p[i][0], p[i][1]),xytext=(p[i][0]+0.1,p[i][1]+0.1))
    pylab.title ("Начальное значение точек")

p = transformation_plot(p, array)

for i in range(len(p)):
    pylab.subplot (2, 1, 2)
    plt.plot(p[i][0], p[i][1],'ko')
    plt.annotate(str(i+1), xy=(p[i][0], p[i][1]),xytext=(p[i][0]+0.2,p[i][1]+0.2))
    pylab.title ("Преобразованное значение точек")

plt.subplots_adjust(hspace=0.5)
plt.show()