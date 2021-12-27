import numpy as np
import matplotlib.pyplot as plt


def arraytransformer(array):
    return (array - array.mean(axis=0)) / array.std(axis=0)
    

a = 7
b = 2

array = np.random.multivariate_normal(
    mean = [1, 2],
    cov = [[2, 1], [1, 3]],
    size = (a)
)

print(array.shape)    
    
plt.scatter(array[:, 0], array[:, 1], c='r')
for i in range(0,a):
    plt.annotate(i + 1, xy=(array[i][0]+0.1, array[i][1]+0.1))
    

array=arraytransformer(array)

plt.scatter(array[:, 0], array[:,1], c='b')
for i in range(0,a):
    plt.annotate(i + 1, xy=(array[i][0]+0.1, array[i][1]+0.1))

plt.show()