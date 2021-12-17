import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


    # пункт а

n = np.random.normal(size=100)
bn = 10 
plt.ylabel('Частота')
plt.xlabel('Данные')


plt.hist(n, density=True, bins = bn)
a = np.linspace(-5.5, 5.5, num=100)
plt.plot(a, stats.norm.pdf(a, 0, 1))
plt.show()
print("Среднее = ",np.mean(n),"при исходном 0")
print("Отклонение = ",np.std(n),"при исходном 1")


#пункт б

ag= 0
ag1  = 0
bd = 0
bd1= 0
cN = 0
cN1 = 0

for i in range(0,150):
    dv = np.random.normal(size=20)
    k = np.var(dv,ddof=0)
    m = np.var(dv,ddof=1)
    
    if m > 1: ag += 1
    if k > 1: ag1 += 1
    if k < 1: bd += 1
    if m < 1: bd1 += 1
    
    ms = (ms+(1-k)**2)/(i+1)
    ms1 = (ms1+(1-m)**2)/(i+1)

print('Количество раз когда дисперсия превысила реальную: ', ag)
print('Количество раз когда исправленная дисперсия превысила реальную: ', ag1)

print('Количество раз когда дисперсия недооценила реальную: ', bd)
print('Количество раз когда исправленная дисперсия недооценила реальную: ', bd1)

print('Средней квадрат ошибки дисперсии', ms)
print('Средней квадрат ошибки исправленной дисперсии', ms1)