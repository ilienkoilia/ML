import numpy as np
import matplotlib.pyplot as plt

def Calculatorf(a):
    return a - (a**3)/6 + (a**5)/120 - (a**7)/5040

def Calculatorg(a):
    return np.sin(a)


amin = -5.0
amax = 5.0
alist = np.linspace(amin, amax, 1000)
blist_f = Calculatorf(alist)
blist_g = Calculatorg(alist)
plt.plot(alist, blist_f)
plt.plot(alist, blist_g)
plt.show()