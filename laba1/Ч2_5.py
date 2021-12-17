import numpy as np
import matplotlib.pyplot as plt



a = np.arange(-5, 5, 0.005)

def function(a):
    return np.sin(a)

p=[]

for ai in a:
    temp = np.array([ai, (function(ai)-function(ai-0.01))/0.01])
    p.append(temp)

plt.plot(a, function(a))    

plt.title("График")
plt.grid(True)
plt.show()

for pt in p:
        plt.plot(pt[0], pt[1], 'r.')

plt.title("Производная")
plt.grid(True)
plt.show()