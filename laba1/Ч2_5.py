import numpy as np
import matplotlib.pyplot as plt



a = np.arange(-5, 5, 0.005)

def function(a):
    return np.sin(a)

p=[]

p = (function(a)-function(a-0.01))/0.01

plt.plot(a, function(a), label="Функция")    
plt.plot(a, np.cos(a), 'b', label="Аналитическая производная")
plt.title("График и аналитическая производная")
plt.grid(True)
plt.legend()

plt.show()

plt.plot(a, p, 'r', label="Разностная производная")
plt.title("Производная")
plt.grid(True)
plt.legend()
plt.show()